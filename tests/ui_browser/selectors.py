"""DOM ids the browser tests rely on. tests/ui/test_selectors.py checks they all exist in index.html,
so renaming an id in the page fails a fast unit test (no browser needed) instead of a confusing browser timeout."""
IDS = [
    "tab-identify", "tab-history", "btn-about", "in-file", "in-camera", "btn-webcam", "btn-snap", "btn-webcam-cancel",
    "video", "webcam", "capture-err", "s-capture", "s-analyzing", "s-result", "s-saved", "res-img", "res-cands",
    "res-warn", "res-notes", "btn-discard", "btn-again", "result-err", "view-identify", "view-history", "flt",
    "hist-count", "hist-list", "btn-more", "hist-empty", "dlg-detail", "d-img", "d-title", "d-sub", "d-cands",
    "d-notes", "d-verify", "d-save", "d-delete", "d-close", "d-err", "dlg-about", "about-body", "about-close",
]
