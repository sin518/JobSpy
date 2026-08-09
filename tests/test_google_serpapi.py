from datetime import UTC, datetime
from unittest import TestCase
from unittest.mock import patch

from jobspy import scrape_jobs
from jobspy.google import Google, GoogleJobsUnavailableError
from jobspy.google.serpapi import SerpApiGoogleJobsClient, SerpApiGoogleJobsError
from jobspy.model import JobResponse, ScraperInput, Site


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

    def get(self, url, *, params, timeout):
        self.requests.append((url, params, timeout))
        return self.response


class _LegacyResponse:
    text = '<noscript><meta content="0;url=/httpservice/retry/enablejs?sei=test"></noscript>'


class _LegacySession:
    def get(self, url, *, headers, params):
        return _LegacyResponse()


def _query(**changes):
    values = {
        "site_type": [Site.GOOGLE],
        "search_term": "Full Stack Developer",
        "location": "Singapore",
        "results_wanted": 10,
        "hours_old": 168,
    }
    values.update(changes)
    return ScraperInput(**values)


class SerpApiGoogleJobsClientTests(TestCase):
    def test_maps_one_bounded_search_without_exposing_provider_types(self):
        session = _FakeSession(
            _FakeResponse(
                {
                    "jobs_results": [
                        {
                            "job_id": "google-job-1",
                            "title": "Full Stack Developer",
                            "company_name": "Example Pte Ltd",
                            "location": "Singapore",
                            "description": "Build customer-facing products.",
                            "detected_extensions": {
                                "posted_at": "2 days ago",
                                "schedule_type": "Full-time",
                                "work_from_home": False,
                            },
                            "apply_options": [
                                {
                                    "title": "Employer",
                                    "link": "https://jobs.example.test/full-stack-1",
                                }
                            ],
                            "share_link": "https://www.google.com/search?job=google-job-1",
                        }
                    ]
                }
            )
        )
        client = SerpApiGoogleJobsClient(
            api_key="test-secret",
            session=session,
            now=lambda: datetime(2026, 8, 9, tzinfo=UTC),
        )

        response = client.scrape(_query())

        self.assertEqual(len(response.jobs), 1)
        job = response.jobs[0]
        self.assertEqual(job.id, "go-google-job-1")
        self.assertEqual(job.title, "Full Stack Developer")
        self.assertEqual(job.company_name, "Example Pte Ltd")
        self.assertEqual(job.job_url, "https://jobs.example.test/full-stack-1")
        self.assertEqual(job.date_posted.isoformat(), "2026-08-07")
        self.assertEqual([item.name for item in job.job_type], ["FULL_TIME"])
        self.assertFalse(job.is_remote)
        self.assertEqual(len(session.requests), 1)
        url, params, timeout = session.requests[0]
        self.assertEqual(url, "https://serpapi.com/search.json")
        self.assertEqual(params["engine"], "google_jobs")
        self.assertEqual(params["api_key"], "test-secret")
        self.assertEqual(params["q"], "Full Stack Developer in the last week")
        self.assertEqual(params["location"], "Singapore")
        self.assertEqual(timeout, 60)

    def test_rejects_an_untrusted_error_payload_without_echoing_it(self):
        session = _FakeSession(
            _FakeResponse({"error": "invalid key: test-secret"}, status_code=401)
        )
        client = SerpApiGoogleJobsClient(api_key="test-secret", session=session)

        with self.assertRaisesRegex(
            SerpApiGoogleJobsError, "Google Jobs provider request failed"
        ) as error:
            client.scrape(_query())

        self.assertNotIn("test-secret", str(error.exception))

    def test_rejects_a_success_payload_without_an_explicit_jobs_collection(self):
        session = _FakeSession(
            _FakeResponse({"search_metadata": {"status": "Success"}})
        )
        client = SerpApiGoogleJobsClient(api_key="test-secret", session=session)

        with self.assertRaisesRegex(
            SerpApiGoogleJobsError, "Google Jobs provider response was invalid"
        ):
            client.scrape(_query())


class ScrapeJobsGoogleProviderTests(TestCase):
    def test_keeps_the_existing_positional_argument_order(self):
        scraper_inputs = []

        class _FakeGoogle:
            def __init__(self, **kwargs):
                pass

            def scrape(self, scraper_input):
                scraper_inputs.append(scraper_input)
                return JobResponse(jobs=[])

        with patch("jobspy.Google", _FakeGoogle):
            scrape_jobs("google", "term", "google term", "Singapore", 25, True)

        self.assertEqual(scraper_inputs[0].location, "Singapore")
        self.assertEqual(scraper_inputs[0].distance, 25)
        self.assertTrue(scraper_inputs[0].is_remote)

    def test_forwards_the_serpapi_key_only_to_the_google_scraper(self):
        constructor_arguments = []

        class _FakeGoogle:
            def __init__(self, **kwargs):
                constructor_arguments.append(kwargs)

            def scrape(self, scraper_input):
                return JobResponse(jobs=[])

        with patch("jobspy.Google", _FakeGoogle):
            scrape_jobs(
                site_name="google",
                search_term="Full Stack Developer",
                serp_api_key="test-secret",
            )

        self.assertEqual(constructor_arguments[0]["serp_api_key"], "test-secret")

    def test_legacy_javascript_shell_is_an_explicit_failure_not_a_false_zero(self):
        with patch("jobspy.google.create_session", return_value=_LegacySession()):
            with self.assertRaisesRegex(
                GoogleJobsUnavailableError, "unavailable without JavaScript"
            ):
                Google().scrape(_query())
