from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Protocol
from urllib.parse import urlparse

from jobspy.model import JobPost, JobResponse, JobType, Location, ScraperInput
from jobspy.util import create_session, extract_emails_from_text, extract_job_type

SERPAPI_GOOGLE_JOBS_URL = "https://serpapi.com/search.json"


class _Response(Protocol):
    status_code: int

    def json(self) -> object: ...


class _Session(Protocol):
    def get(
        self, url: str, *, params: Mapping[str, object], timeout: int
    ) -> _Response: ...


class SerpApiGoogleJobsError(RuntimeError):
    """A credential-safe failure at the optional Google Jobs provider boundary."""


class SerpApiGoogleJobsClient:
    """Fetch one bounded Google Jobs result page without exposing provider payloads."""

    def __init__(
        self,
        *,
        api_key: str,
        session: _Session | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if (
            not isinstance(api_key, str)
            or not api_key
            or any(character in api_key for character in "\r\n\x00")
        ):
            raise SerpApiGoogleJobsError(
                "Google Jobs provider credentials are invalid."
            )
        self._api_key = api_key
        self._session = session or create_session(is_tls=False, has_retry=True)
        self._now = now

    def scrape(self, scraper_input: ScraperInput) -> JobResponse:
        params: dict[str, object] = {
            "api_key": self._api_key,
            "engine": "google_jobs",
            "google_domain": "google.com",
            "hl": "en",
            "q": _search_query(scraper_input),
        }
        if scraper_input.location:
            params["location"] = scraper_input.location
        try:
            response = self._session.get(
                SERPAPI_GOOGLE_JOBS_URL,
                params=params,
                timeout=scraper_input.request_timeout,
            )
            payload = response.json()
        except Exception:
            raise SerpApiGoogleJobsError(
                "Google Jobs provider request failed."
            ) from None
        if not 200 <= response.status_code < 300:
            raise SerpApiGoogleJobsError("Google Jobs provider request failed.")
        if not isinstance(payload, Mapping) or "error" in payload:
            raise SerpApiGoogleJobsError("Google Jobs provider request failed.")
        raw_jobs = payload.get("jobs_results")
        if not isinstance(raw_jobs, Sequence) or isinstance(raw_jobs, str | bytes):
            raise SerpApiGoogleJobsError("Google Jobs provider response was invalid.")

        start = scraper_input.offset
        end = start + scraper_input.results_wanted
        observed_at = self._now()
        if observed_at.tzinfo is None:
            raise SerpApiGoogleJobsError("Google Jobs provider clock is invalid.")
        jobs = [_job_post(raw_job, observed_at) for raw_job in raw_jobs[start:end]]
        return JobResponse(jobs=jobs)


def _search_query(scraper_input: ScraperInput) -> str:
    if scraper_input.google_search_term:
        return scraper_input.google_search_term.strip()
    if not scraper_input.search_term or not scraper_input.search_term.strip():
        raise SerpApiGoogleJobsError("Google Jobs search term is invalid.")

    parts = [scraper_input.search_term.strip()]
    if scraper_input.job_type is not None:
        labels = {
            JobType.FULL_TIME: "Full time",
            JobType.PART_TIME: "Part time",
            JobType.INTERNSHIP: "Internship",
            JobType.CONTRACT: "Contract",
        }
        label = labels.get(scraper_input.job_type)
        if label:
            parts.append(label)
    if scraper_input.hours_old is not None:
        if scraper_input.hours_old <= 24:
            parts.append("since yesterday")
        elif scraper_input.hours_old <= 72:
            parts.append("in the last 3 days")
        elif scraper_input.hours_old <= 168:
            parts.append("in the last week")
        else:
            parts.append("in the last month")
    if scraper_input.is_remote:
        parts.append("remote")
    return " ".join(parts)


def _job_post(raw_job: object, now: datetime) -> JobPost:
    if not isinstance(raw_job, Mapping):
        raise SerpApiGoogleJobsError("Google Jobs provider response was invalid.")
    job_id = _required_text(raw_job, "job_id")
    title = _required_text(raw_job, "title")
    company = _required_text(raw_job, "company_name")
    description = _optional_text(raw_job.get("description"))
    location = _optional_text(raw_job.get("location"))
    extensions = raw_job.get("detected_extensions")
    if extensions is None:
        extensions = {}
    if not isinstance(extensions, Mapping):
        raise SerpApiGoogleJobsError("Google Jobs provider response was invalid.")
    schedule_type = _optional_text(extensions.get("schedule_type"))
    work_from_home = extensions.get("work_from_home")
    if work_from_home is not None and not isinstance(work_from_home, bool):
        raise SerpApiGoogleJobsError("Google Jobs provider response was invalid.")
    remote = (
        bool(work_from_home)
        or "remote"
        in " ".join(value for value in (location, description) if value).casefold()
    )

    return JobPost(
        id=f"go-{job_id}",
        title=title,
        company_name=company,
        location=Location(city=location) if location else None,
        job_url=_job_url(raw_job),
        date_posted=_posted_date(_optional_text(extensions.get("posted_at")), now),
        is_remote=remote,
        description=description,
        emails=extract_emails_from_text(description),
        job_type=extract_job_type(
            (schedule_type or description or "").replace("-", " ")
        ),
    )


def _job_url(raw_job: Mapping[str, object]) -> str:
    apply_options = raw_job.get("apply_options")
    if apply_options is not None:
        if not isinstance(apply_options, Sequence) or isinstance(
            apply_options, str | bytes
        ):
            raise SerpApiGoogleJobsError("Google Jobs provider response was invalid.")
        for option in apply_options:
            if isinstance(option, Mapping):
                link = _optional_text(option.get("link"))
                if link and _is_http_url(link):
                    return link
    share_link = _optional_text(raw_job.get("share_link"))
    if share_link and _is_http_url(share_link):
        return share_link
    raise SerpApiGoogleJobsError("Google Jobs provider response was invalid.")


def _posted_date(value: str | None, now: datetime) -> date | None:
    if value is None:
        return None
    normalized = value.casefold()
    if normalized in {"today", "just posted"}:
        return now.astimezone(UTC).date()
    if normalized == "yesterday":
        return now.astimezone(UTC).date() - timedelta(days=1)
    match = re.search(r"(\d+)\+?\s+(hour|day|week|month)s?\s+ago", normalized)
    if match is None:
        return None
    amount = int(match.group(1))
    unit_days = {"hour": 0, "day": 1, "week": 7, "month": 30}
    return now.astimezone(UTC).date() - timedelta(
        days=amount * unit_days[match.group(2)]
    )


def _required_text(values: Mapping[str, object], key: str) -> str:
    value = _optional_text(values.get(key))
    if value is None:
        raise SerpApiGoogleJobsError("Google Jobs provider response was invalid.")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SerpApiGoogleJobsError("Google Jobs provider response was invalid.")
    normalized = " ".join(value.split())
    return normalized or None


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
