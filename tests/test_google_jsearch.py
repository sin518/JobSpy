from datetime import date
from hashlib import sha256
from unittest import TestCase
from unittest.mock import patch

from jobspy import scrape_jobs
from jobspy.google.jsearch import JSearchGoogleJobsClient, JSearchGoogleJobsError
from jobspy.model import Country, JobPost, JobResponse, Location, ScraperInput, Site

LONG_JOB_ID = "google-job-" + ("x" * 400)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def get(self, url, *, params, headers, timeout):
        self.requests.append((url, params, headers, timeout))
        return self.response


def _query(**changes):
    values = {
        "site_type": [Site.GOOGLE],
        "search_term": "Full Stack Developer",
        "location": "Singapore",
        "results_wanted": 10,
        "hours_old": 168,
        "country": Country.SINGAPORE,
    }
    values.update(changes)
    return ScraperInput(**values)


def _raw_job(**changes):
    values = {
        "job_id": "google-job-1",
        "job_title": "Full Stack Developer",
        "employer_name": "Example Pte Ltd",
        "job_publisher": "JobStreet",
        "job_employment_type": "Full-time",
        "job_apply_link": "https://jobs.example.test/full-stack-1",
        "job_description": "Build customer-facing products.",
        "job_is_remote": False,
        "job_posted_at_datetime_utc": "2026-08-07T12:30:00.000Z",
        "job_location": "Singapore",
        "job_city": "Singapore",
        "job_country": "SG",
    }
    values.update(changes)
    return values


def _client_for(raw_job):
    payload = {"status": "OK", "data": {"jobs": [raw_job]}}
    return JSearchGoogleJobsClient(
        api_key="test-secret", session=_FakeSession(_FakeResponse(payload))
    )


class JSearchGoogleJobsClientTests(TestCase):
    def test_disables_transport_retries_to_keep_quota_usage_bounded(self):
        session = _FakeSession(_FakeResponse({"status": "OK", "data": {"jobs": []}}))

        with patch(
            "jobspy.google.jsearch.create_session", return_value=session
        ) as factory:
            JSearchGoogleJobsClient(api_key="test-secret").scrape(_query())

        factory.assert_called_once_with(is_tls=False, has_retry=False)

    def test_maps_one_bounded_search_and_keeps_the_key_in_the_header(self):
        session = _FakeSession(
            _FakeResponse(
                {
                    "status": "OK",
                    "data": {
                        "jobs": [
                            {
                                "job_id": LONG_JOB_ID,
                                "job_title": "Full Stack Developer",
                                "employer_name": "Example Pte Ltd",
                                "employer_website": "https://example.test",
                                "job_publisher": "Indeed",
                                "job_employment_type": "Full-time",
                                "job_apply_link": "https://jobs.example.test/full-stack-1",
                                "job_description": "Build customer-facing products.",
                                "job_is_remote": False,
                                "job_posted_at_datetime_utc": "2026-08-07T12:30:00.000Z",
                                "job_location": "Singapore",
                                "job_city": "Singapore",
                                "job_country": "SG",
                                "job_min_salary": 7000,
                                "job_max_salary": 9000,
                                "job_salary_period": "MONTH",
                                "job_salary_currency": "SGD",
                            }
                        ],
                        "cursor": "do-not-follow-in-the-bounded-client",
                    },
                }
            )
        )
        client = JSearchGoogleJobsClient(api_key="test-secret", session=session)

        response = client.scrape(_query())

        self.assertEqual(len(response.jobs), 1)
        job = response.jobs[0]
        self.assertEqual(
            job.id,
            f"go-{sha256(LONG_JOB_ID.encode('utf-8')).hexdigest()}",
        )
        self.assertEqual(len(job.id), 67)
        self.assertEqual(job.title, "Full Stack Developer")
        self.assertEqual(job.company_name, "Example Pte Ltd")
        self.assertEqual(job.job_url, "https://jobs.example.test/full-stack-1")
        self.assertEqual(job.company_url, "https://example.test")
        self.assertEqual(job.date_posted, date(2026, 8, 7))
        self.assertEqual(job.location.city, "Singapore")
        self.assertEqual(job.location.country, "SG")
        self.assertEqual([item.name for item in job.job_type], ["FULL_TIME"])
        self.assertFalse(job.is_remote)
        self.assertEqual(job.compensation.min_amount, 7000)
        self.assertEqual(job.compensation.max_amount, 9000)
        self.assertEqual(job.compensation.currency, "SGD")
        self.assertEqual(job.compensation.interval.value, "monthly")

        self.assertEqual(len(session.requests), 1)
        url, params, headers, timeout = session.requests[0]
        self.assertEqual(url, "https://api.openwebninja.com/jsearch/search-v2")
        self.assertEqual(params["query"], "Full Stack Developer jobs in Singapore")
        self.assertEqual(params["country"], "sg")
        self.assertEqual(params["language"], "en")
        self.assertEqual(params["date_posted"], "week")
        self.assertEqual(params["num_pages"], 1)
        self.assertEqual(
            headers, {"x-api-key": "test-secret", "Accept": "application/json"}
        )
        self.assertNotIn("api_key", params)
        self.assertEqual(timeout, 60)

    def test_structured_compensation_prefers_an_explicit_valid_currency(self):
        client = _client_for(
            _raw_job(
                job_min_salary=4700,
                job_max_salary=6000,
                job_salary_period="MONTH",
                job_salary_currency="EUR",
            )
        )

        job = client.scrape(_query()).jobs[0]

        self.assertEqual(job.compensation.currency, "EUR")

    def test_structured_compensation_uses_an_exact_country_currency_when_missing(self):
        client = _client_for(
            _raw_job(
                job_min_salary=4700,
                job_max_salary=6000,
                job_salary_period="MONTH",
                job_salary_currency=None,
                job_country="SG",
            )
        )

        job = client.scrape(_query()).jobs[0]

        self.assertEqual(job.compensation.currency, "SGD")

    def test_structured_compensation_does_not_invent_currency_for_unknown_country(self):
        client = _client_for(
            _raw_job(
                job_min_salary=4700,
                job_max_salary=6000,
                job_salary_period="MONTH",
                job_salary_currency=None,
                job_country="ZZ",
            )
        )

        job = client.scrape(_query()).jobs[0]

        self.assertIsNone(job.compensation.currency)

    def test_keeps_highlight_salary_separate_from_the_job_description(self):
        description = "Build customer-facing products."
        client = _client_for(
            _raw_job(
                job_description=description,
                job_highlights={
                    "Benefits": [
                        "$4,700 - $6,000 per month",
                        "Medical and dental insurance",
                    ]
                },
            )
        )

        job = client.scrape(_query()).jobs[0]

        self.assertEqual(job.salary_text, "$4,700 - $6,000 per month")
        self.assertEqual(job.description, description)
        self.assertNotIn(job.salary_text, job.description)

    def test_ignores_non_base_pay_financial_highlights(self):
        candidates = (
            "Transport allowance: S$500 per month",
            "Annual performance bonus up to S$10,000",
            "Raised $286M in funding",
        )
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                client = _client_for(_raw_job(job_highlights={"Benefits": [candidate]}))

                job = client.scrape(_query()).jobs[0]

                self.assertIsNone(job.salary_text)

    def test_ignores_malformed_or_oversized_highlights_without_leaking_payloads(self):
        secret = "do-not-echo-highlight-payload"
        malformed_values = (
            [secret],
            {"Benefits": secret},
            {"Benefits": [secret + ("x" * 5000)]},
            {secret + ("x" * 100): ["$4,700 - $6,000 per month"]},
            {"Benefits": ["$4,700 - $6,000 per month\x00" + secret]},
        )
        for highlights in malformed_values:
            with self.subTest(highlights_type=type(highlights).__name__):
                client = _client_for(_raw_job(job_highlights=highlights))

                job = client.scrape(_query()).jobs[0]

                self.assertIsNone(job.salary_text)

    def test_maps_date_ranges_to_the_supported_jsearch_values(self):
        cases = (
            (None, "all"),
            (24, "today"),
            (72, "3days"),
            (168, "week"),
            (720, "month"),
        )
        for hours_old, expected in cases:
            with self.subTest(hours_old=hours_old):
                session = _FakeSession(
                    _FakeResponse({"status": "OK", "data": {"jobs": []}})
                )
                JSearchGoogleJobsClient(api_key="test-secret", session=session).scrape(
                    _query(hours_old=hours_old)
                )
                self.assertEqual(session.requests[0][1]["date_posted"], expected)

    def test_rejects_untrusted_failures_without_echoing_credentials_or_payloads(self):
        secret = "test-secret"
        session = _FakeSession(
            _FakeResponse(
                {"status": "ERROR", "message": f"invalid key: {secret}"},
                status_code=401,
            )
        )

        with self.assertRaisesRegex(
            JSearchGoogleJobsError, "Google Jobs provider request failed"
        ) as error:
            JSearchGoogleJobsClient(api_key=secret, session=session).scrape(_query())

        self.assertNotIn(secret, str(error.exception))

    def test_rejects_a_success_payload_without_an_explicit_jobs_collection(self):
        session = _FakeSession(_FakeResponse({"status": "OK", "data": []}))

        with self.assertRaisesRegex(
            JSearchGoogleJobsError, "Google Jobs provider response was invalid"
        ):
            JSearchGoogleJobsClient(api_key="test-secret", session=session).scrape(
                _query()
            )


class ScrapeJobsJSearchProviderTests(TestCase):
    def test_forwards_the_jsearch_key_only_to_the_google_scraper(self):
        constructor_arguments = []
        scraper_inputs = []

        class _FakeGoogle:
            def __init__(self, **kwargs):
                constructor_arguments.append(kwargs)

            def scrape(self, scraper_input):
                scraper_inputs.append(scraper_input)
                return JobResponse(jobs=[])

        with patch("jobspy.Google", _FakeGoogle):
            scrape_jobs(
                site_name="google",
                search_term="Full Stack Developer",
                country_indeed="singapore",
                jsearch_api_key="test-secret",
            )

        self.assertEqual(constructor_arguments[0]["jsearch_api_key"], "test-secret")
        self.assertEqual(scraper_inputs[0].country, Country.SINGAPORE)

    def test_exposes_salary_text_as_an_optional_dataframe_column(self):
        class _FakeGoogle:
            def __init__(self, **kwargs):
                pass

            def scrape(self, scraper_input):
                return JobResponse(
                    jobs=[
                        JobPost(
                            id="google-job-1",
                            title="Full Stack Developer",
                            company_name="Example Pte Ltd",
                            job_url="https://jobs.example.test/full-stack-1",
                            location=Location(city="Singapore", country="SG"),
                            description="Build customer-facing products.",
                            salary_text="$4,700 - $6,000 per month",
                        )
                    ]
                )

        with patch("jobspy.Google", _FakeGoogle):
            jobs = scrape_jobs(
                site_name="google",
                search_term="Full Stack Developer",
                country_indeed="singapore",
                jsearch_api_key="test-secret",
            )

        self.assertIn("salary_text", jobs.columns)
        self.assertEqual(jobs.iloc[0]["salary_text"], "$4,700 - $6,000 per month")
        self.assertEqual(jobs.iloc[0]["description"], "Build customer-facing products.")
