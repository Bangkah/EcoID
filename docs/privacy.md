# Privacy

Implements `SRS.md` NFR-003 and section 11.

- Identification runs locally (ONNX Runtime, CPU). The app contains no code that opens outbound connections.
- Photos, notes and observations live only in the data directory on this device.
- The server listens on `127.0.0.1` by default and rejects requests whose `Host` header is not a loopback name
  (protects against DNS-rebinding from web pages you visit). `--host 0.0.0.0` exposes the app, which has **no login**,
  to your whole network — use only on a network you trust.
- The page loads no external fonts, scripts, styles or images; a strict Content-Security-Policy makes the browser
  enforce that.
- **Location is opt-in, per observation.** Nothing is attached unless you tap *Use the photo's location*, *Use my current location*, type coordinates, or tick *Always use the photo's location*.
  Only coordinates, their source (`exif`/`device`/`manual`) and accuracy are stored. EcoID itself sends nothing anywhere.
- *Use my current location* calls the browser's Geolocation API. On a phone that is GPS and works offline. On a laptop, some browsers (e.g. Chrome) look the position up through an online
  service — that request is made by the browser, outside EcoID. If that matters, type coordinates or use the photo's own EXIF position instead.
- Users can delete any observation (row, photo and thumbnail).
- EXIF metadata inside photos (which may include GPS from the phone camera) is preserved in the stored original,
  because the original is kept untouched for verification — **even if you did not attach the location to the observation**. Strip it before sharing a photo.
  `scripts/export_observations.py --strip-exif` removes it for photos you export into the dataset (orientation is preserved).

- **The map never loads tiles.** Observations are drawn locally on a coordinate grid; an optional background is a picture *you* put in the data folder.
- **Export is you deciding to move data out.** CSV/JSON/ZIP can include locations (switch: "Include locations"); a backup ZIP contains the original photos with their EXIF
  (which may hold GPS even if you turned the switch off). Share exports with the same care as the photos. GeoJSON always contains locations.

## Verified by tests
`tests/ui/test_app_e2e.py` runs the whole flow with all non-loopback sockets and DNS blocked, and fails if any
outbound attempt is recorded; it also checks the page has no external references.
