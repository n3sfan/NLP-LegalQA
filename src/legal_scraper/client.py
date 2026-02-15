"""API client for phapluat.gov.vn legal document endpoints."""

from __future__ import annotations

import urllib3

import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://phapluat.gov.vn/api/legal-documents"

DEFAULT_HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "origin": "https://phapluat.gov.vn",
    "referer": "https://phapluat.gov.vn/he-thong-van-ban-phap-luat",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    ),
}


class LegalDocumentClient:
    """Low-level HTTP client wrapping the phapluat.gov.vn API."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def search(
        self,
        keywords: str = "",
        *,
        date_from: str = "01/01/1945",
        date_to: str = "15/02/2026",
        page_index: int = 0,
        row_amount: int = 100,
        is_search_exact: int = 0,
        search_options: int = 1,
        search_by_date: str = "crDateTime",
        sort_by: str = "crDateTime",
        sort_order: str = "desc",
        doc_group_ids: list[int] | None = None,
        field_ids: list[int] | None = None,
        effect_status_ids: list[int] | None = None,
        signer_ids: list[int] | None = None,
        organ_ids: list[int] | None = None,
        doc_type_ids: list[int] | None = None,
        language_id: int = 1,
    ) -> dict:
        """Search for legal documents. Returns the full API response dict."""
        # PLACEHOLDER_SEARCH_BODY
        body = {
            "keywords": keywords,
            "isSearchExact": is_search_exact,
            "dateFrom": date_from,
            "dateTo": date_to,
            "pageIndex": page_index,
            "rowAmount": row_amount,
            "searchOptions": search_options,
            "searchByDate": search_by_date,
            "sortBy": sort_by,
            "sortOrder": sort_order,
            "docGroupIds": doc_group_ids or [],
            "fieldIds": field_ids or [],
            "effectStatusIds": effect_status_ids or [],
            "signerIds": signer_ids or [],
            "organIds": organ_ids or [],
            "docTypeIds": doc_type_ids or [],
            "languageId": language_id,
        }
        resp = self.session.post(BASE_URL, json=body, verify=False)
        resp.raise_for_status()
        return resp.json()

    def get_detail(self, doc_guid: str, tab: str = "noidung") -> dict:
        """Fetch document detail (tab: 'tomtat' or 'noidung')."""
        resp = self.session.get(
            f"{BASE_URL}/detail",
            params={"docGUId": doc_guid, "tabName": tab},
            verify=False,
        )
        resp.raise_for_status()
        return resp.json()
