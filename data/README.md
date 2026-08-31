# Local input data and caches

Place `JPL_CR3BP_OrbitCatalog.mat` here. An existing copy at the project root
continues to work if there is no catalog in this directory.

Optional raw orbit CSV files belong in `JPL_Data/`. Only run
`scripts/load_and_filter_data.m` when you intend to rebuild the catalog.

Generated caches are written to `cache/orbits/` and `cache/transfers/`.
These folders are created when needed. Old root-level caches are not reused
automatically, and no existing data are moved or deleted by setup.

Everything here except this README is ignored by Git.
