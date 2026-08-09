from datetime import date
from unittest import TestCase
from unittest.mock import patch

from jobspy import scrape_jobs
from jobspy.google.jsearch import JSearchGoogleJobsClient, JSearchGoogleJobsError
from jobspy.model import Country, JobResponse, ScraperInput, Site


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


class JSearchGoogleJobsClientTests(TestCase):
    def test_maps_one_bounded_search_and_keeps_the_key_in_the_header(self):
        session = _FakeSession(
            _FakeResponse(
                {
                    "status": "OK",
                    "data": {
                        "jobs": [
                            {
                                "job_id": "google-job-1",
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
        self.assertEqual(job.id, "go-google-job-1")
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
