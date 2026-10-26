from __future__ import annotations

import unittest

from scripts.audit_llm_credentials import _classify_http_error


class CredentialAuditTests(unittest.TestCase):
    def test_authentication_failure_is_invalid(self) -> None:
        self.assertEqual(_classify_http_error(401, "Unauthorized"), "invalid_auth")
        self.assertEqual(
            _classify_http_error(403, "Authentication error"),
            "invalid_auth",
        )

    def test_quota_failure_is_not_invalid(self) -> None:
        self.assertEqual(
            _classify_http_error(429, "Tokens per day quota reached"),
            "quota_exhausted",
        )

    def test_generic_permission_failure_is_kept(self) -> None:
        self.assertEqual(
            _classify_http_error(403, "Model access is forbidden"),
            "permission_denied",
        )


if __name__ == "__main__":
    unittest.main()
