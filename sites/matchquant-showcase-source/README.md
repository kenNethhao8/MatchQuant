# MatchQuant v2 Showcase

A buildless, single-file presentation site for MatchQuant v2. The public page is `dist/index.html`; markup, styles, interaction logic, and the documented Chelsea–Man United sample are embedded in that file.

## Preview

Serve the static directory locally:

```powershell
python -m http.server 4178 --directory dist
```

Then open <http://localhost:4178>. No package install, Python application environment, prediction backend, or API key is required to render the sample or use the local demo controls.

## Static data and media

- `SAMPLE_CHELSEA_MAN_UNITED_DATA` is embedded from section 3 of `docs/MATCHQUANT_V2_SHOWCASE_AND_MARKETING_GUIDE.md`.
- The source object contains four explicit timeline timestamps. The report describes seven stable observations, so the chart adds three clearly labeled carry-forward display nodes without inventing quote changes or exact source timestamps.
- The smoke still and loop video use the CDN URLs supplied in the design brief. The still remains visible when media is unavailable; reduced-motion mode hides the video.
- `scripts/build_showcase_data.py` remains as the archived V1 research exporter. The v2 page does not use it, and its stale generated JSON is no longer shipped in `dist/`.

## Research boundary

The site presents a static research example. It does not run a forecast, call an odds API, or provide betting advice. Historical performance and test counts are claims recorded in the project guide, not live measurements made by the page.

