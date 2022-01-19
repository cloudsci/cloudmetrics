
mask-based:

- `cloud_fraction` (previously `cf`)
  - `cloud_mask`
- `fractal_dimension` (previously `fracDim`)
  - `cloud_mask`
- `open_sky`
  - `cloud_mask`
- `orientation`
  - `cloud_mask`

object-based:

- `cop`
  - Primary: `cloud_mask`
  - Derived: `labelled_image` -> `region_properties`
- `csd`
  - Primary: `cloud_mask`
  - Derived: `labelled_image` -> `region_properties`
- `iorg`
  - Primary: `cloud_mask`
  - Derived: `labelled_image` -> `region_properties`
- `iorg_poisson`
  - Primary: `cloud_mask`
  - Derived: `labelled_image` -> `region_properties`
- `objects`
  - Primary: `cloud_mask`
  - Derived: `labelled_image` -> `region_properties`
- `rdf`
  - Primary: `cloud_mask`
  - Derived: `labelled_image` -> `region_properties`
- `scai`
  - Primary: `cloud_mask`
  - Derived: `labelled_image` -> `region_properties`

scalar-based:

- `cth`
  - `cloud_top_height` scalar field
- `cwp`
  - `cloud_water_path` scalar field
- `fourier`
  - Primary: Any `cloud_scalar` or `cloud_mask` would work (I've used `cloud_water_path` so far)
  - Derived: Radially and azimuthally averaged power spectra of the primary input
- `woi`
  - Primary: `cloud_scalar`
  - Derived: Scale-decomposed stationary wavelet spectra (in vertical, horizontal and diagonal directions)

other:

- `network`
  - Will defer incorporating network metrics until we know a little more about what they're good at.
