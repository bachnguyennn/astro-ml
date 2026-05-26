# Catalogs

## HYG star catalog (required)

The synthetic data renderer needs the HYG v3 catalog
(119,614 stars with RA, Dec, magnitude, color index).

```bash
# from the repo root:
mkdir -p data/catalogs
curl -L -o data/catalogs/hygdata_v3.csv \
  https://raw.githubusercontent.com/astronexus/HYG-Database/master/hyg/v3/hyg_v36.csv
```

If that URL moves, mirror: https://github.com/astronexus/HYG-Database

The CSV is ~8 MB; it is **gitignored** because it changes upstream and we
want a reproducible explicit download step.

## Constellation lines (optional, for overlay rendering)

```bash
curl -L -o data/catalogs/constellation_lines.json \
  https://raw.githubusercontent.com/Stellarium/stellarium-skycultures/master/modern_iau/constellationship.fab
```
