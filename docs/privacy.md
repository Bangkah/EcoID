# Privacy

Implements `SRS.md` NFR-003 and section 11.

- Identification runs locally (ONNX Runtime, CPU). The app contains no code that opens outbound connections.
- Photos, notes and observations live only in the data directory on this device.
- The server listens on `127.0.0.1` by default and rejects requests whose `Host` header is not a loopback name
  (protects against DNS-rebinding from web pages you visit). `--host 0.0.0.0` exposes the app, which has **no login**,
  to your whole network — use only on a network you trust.
- The page loads no external fonts, scripts, styles or images; a strict Content-Security-Policy makes the browser
  enforce that.
- Location is **not collected** in Phase 3. The fields exist in the record (optional, null by default). When GPS
  capture is added (Eco Mapper), note that some browsers resolve geolocation through an online service; a native
  GPS source is the only fully offline option.
- Users can delete any observation (row, photo and thumbnail).
- EXIF metadata inside photos (which may include GPS from the phone camera) is preserved in the stored original,
  because the original is kept untouched for verification. Strip it before sharing a photo.

## Verified by tests
`tests/ui/test_app_e2e.py` runs the whole flow with all non-loopback sockets and DNS blocked, and fails if any
outbound attempt is recorded; it also checks the page has no external references.
