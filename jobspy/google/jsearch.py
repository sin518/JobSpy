from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Protocol
from urllib.parse import urlparse

from jobspy.model import (
    Compensation,
    CompensationInterval,
    Country,
    JobPost,
    JobResponse,
    JobType,
    Location,
    ScraperInput,
)
from jobspy.util import create_session, extract_emails_from_text, extract_job_type

JSEARCH_GOOGLE_JOBS_URL = "https://api.openwebninja.com/jsearch/search-v2"


class _Response(Protocol):
    status_code: int

    def json(self) -> object: ...


class _Session(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: int,
    ) -> _Response: ...


class JSearchGoogleJobsError(RuntimeError):
    """A credential-safe failure at the optional Google Jobs provider boundary."""


class JSearchGoogleJobsClient:
    """Fetch one JSearch page so callers can predict and bound quota use."""

    def __init__(self, *, api_key: str, session: _Session | None = None) -> None:
        if (
            not isinstance(api_key, str)
            or not api_key
            or any(character.isspace() or character == "\x00" for character in api_key)
        ):
            raise JSearchGoogleJobsError(
                "Google Jobs provider credentials are invalid."
            )
        self._api_key = api_key
        self._session = session or create_session(is_tls=False, has_retry=True)

    def scrape(self, scraper_input: ScraperInput) -> JobResponse:
        params: dict[str, object] = {
            "query": _search_query(scraper_input),
            "country": _country_code(scraper_input.country),
            "language": "en",
            "date_posted": _date_posted(scraper_input.hours_old),
            "num_pages": 1,
        }
        if scraper_input.is_remote:
            params["work_from_home"] = True
        employment_type = _employment_type(scraper_input.job_type)
        if employment_type is not None:
            params["employment_types"] = employment_type

        try:
            response = self._session.get(
                JSEARCH_GOOGLE_JOBS_URL,
                params=params,
                headers={
                    "x-api-key": self._api_key,
                    "Accept": "application/json",
                },
                timeout=scraper_input.request_timeout,
            )
            payload = response.json()
        except Exception:
            raise JSearchGoogleJobsError(
                "Google Jobs provider request failed."
            ) from None
        if not 200 <= response.status_code < 300:
            raise JSearchGoogleJobsError("Google Jobs provider request failed.")
        if not isinstance(payload, Mapping) or payload.get("status") != "OK":
            raise JSearchGoogleJobsError("Google Jobs provider request failed.")
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise JSearchGoogleJobsError("Google Jobs provider response was invalid.")
        raw_jobs = data.get("jobs")
        if not isinstance(raw_jobs, Sequence) or isinstance(raw_jobs, str | bytes):
            raise JSearchGoogleJobsError("Google Jobs provider response was invalid.")

        start = scraper_input.offset
        end = start + scraper_input.results_wanted
        return JobResponse(jobs=[_job_post(raw_job) for raw_job in raw_jobs[start:end]])


def _search_query(scraper_input: ScraperInput) -> str:
    if scraper_input.google_search_term:
        query = " ".join(scraper_input.google_search_term.split())
    elif scraper_input.search_term:
        query = " ".join(scraper_input.search_term.split())
    else:
        query = ""
    if not query:
        raise JSearchGoogleJobsError("Google Jobs search term is invalid.")
    if scraper_input.google_search_term:
        return query
    parts = [query, "jobs"]
    if scraper_input.location and scraper_input.location.strip():
        parts.extend(("in", " ".join(scraper_input.location.split())))
    return " ".join(parts)


def _country_code(country: Country | None) -> str:
    if country is None:
        return "us"
    try:
        _, country_code = country.indeed_domain_value
    except (AttributeError, IndexError, TypeError, ValueError):
        raise JSearchGoogleJobsError("Google Jobs country is invalid.") from None
    normalized = country_code.casefold()
    if len(normalized) != 2 or not normalized.isascii() or not normalized.isalpha():
        raise JSearchGoogleJobsError("Google Jobs country is invalid.")
    return normalized


def _date_posted(hours_old: int | None) -> str:
    if hours_old is None:
        return "all"
    if hours_old <= 24:
        return "today"
    if hours_old <= 72:
        return "3days"
    if hours_old <= 168:
        return "week"
    return "month"


def _employment_type(job_type: JobType | None) -> str | None:
    return {
        JobType.FULL_TIME: "FULLTIME",
        JobType.CONTRACT: "CONTRACTOR",
        JobType.PART_TIME: "PARTTIME",
        JobType.INTERNSHIP: "INTERN",
    }.get(job_type)


def _job_post(raw_job: object) -> JobPost:
    if not isinstance(raw_job, Mapping):
        raise JSearchGoogleJobsError("Google Jobs provider response was invalid.")
    description = _optional_text(raw_job.get("job_description"))
    employment_type = _optional_text(raw_job.get("job_employment_type"))
    company_url = _optional_http_url(raw_job.get("employer_website"))
    remote = raw_job.get("job_is_remote")
    if remote is not None and not isinstance(remote, bool):
        raise JSearchGoogleJobsError("Google Jobs provider response was invalid.")

    return JobPost(
        id=f"go-{_required_text(raw_job, 'job_id')}",
        title=_required_text(raw_job, "job_title"),
        company_name=_required_text(raw_job, "employer_name"),
        company_url=company_url,
        job_url=_job_url(raw_job),
        location=_location(raw_job),
        description=description,
        emails=extract_emails_from_text(description),
        job_type=extract_job_type((employment_type or "").replace("-", " ")),
        compensation=_compensation(raw_job),
        date_posted=_posted_date(raw_job.get("job_posted_at_datetime_utc")),
        is_remote=remote,
        listing_type=_optional_text(raw_job.get("job_publisher")),
    )


def _job_url(raw_job: Mapping[str, object]) -> str:
    direct = _optional_http_url(raw_job.get("job_apply_link"))
    if direct is not None:
        return direct
    apply_options = raw_job.get("apply_options")
    if apply_options is not None:
        if not isinstance(apply_options, Sequence) or isinstance(
            apply_options, str | bytes
        ):
            raise JSearchGoogleJobsError("Google Jobs provider response was invalid.")
        for option in apply_options:
            if isinstance(option, Mapping):
                link = _optional_http_url(option.get("apply_link"))
                if link is not None:
                    return link
    google_link = _optional_http_url(raw_job.get("job_google_link"))
    if google_link is not None:
        return google_link
    raise JSearchGoogleJobsError("Google Jobs provider response was invalid.")


def _location(raw_job: Mapping[str, object]) -> Location | None:
    city = _optional_text(raw_job.get("job_city"))
    state = _optional_text(raw_job.get("job_state"))
    country = _optional_text(raw_job.get("job_country"))
    if city is None and state is None:
        city = _optional_text(raw_job.get("job_location"))
    if city is None and state is None and country is None:
        return None
    return Location(city=city, state=state, country=country)


def _compensation(raw_job: Mapping[str, object]) -> Compensation | None:
    minimum = _optional_number(raw_job.get("job_min_salary"))
    maximum = _optional_number(raw_job.get("job_max_salary"))
    if minimum is None and maximum is None:
        return None
    interval = {
        "YEAR": CompensationInterval.YEARLY,
        "MONTH": CompensationInterval.MONTHLY,
        "WEEK": CompensationInterval.WEEKLY,
        "DAY": CompensationInterval.DAILY,
        "HOUR": CompensationInterval.HOURLY,
    }.get((_optional_text(raw_job.get("job_salary_period")) or "").upper())
    currency = _optional_text(raw_job.get("job_salary_currency")) or "USD"
    return Compensation(
        interval=interval,
        min_amount=minimum,
        max_amount=maximum,
        currency=currency,
    )


def _posted_date(value: object) -> date | None:
    text = _optional_text(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise JSearchGoogleJobsError(
            "Google Jobs provider response was invalid."
        ) from None
    if parsed.tzinfo is None:
        raise JSearchGoogleJobsError("Google Jobs provider response was invalid.")
    return parsed.date()


def _required_text(values: Mapping[str, object], key: str) -> str:
    value = _optional_text(values.get(key))
    if value is None:
        raise JSearchGoogleJobsError("Google Jobs provider response was invalid.")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise JSearchGoogleJobsError("Google Jobs provider response was invalid.")
    normalized = " ".join(value.split())
    return normalized or None


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise JSearchGoogleJobsError("Google Jobs provider response was invalid.")
    number = float(value)
    if not math.isfinite(number):
        raise JSearchGoogleJobsError("Google Jobs provider response was invalid.")
    return number


def _optional_http_url(value: object) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    parsed = urlparse(text)
    return text if parsed.scheme in {"http", "https"} and parsed.netloc else None
