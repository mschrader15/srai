# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2023-MM-DD

### Added

### Changed

- BREAKING: renamed Regionizer to Regionalizer [#282](https://github.com/srai-lab/srai/issues/282)

### Deprecated

### Removed

### Fixed

### Security

## [0.1.1] - 2023-04-27

### Added

- SlippyMapRegionizer
- OSMTileLoader
- GTFS Loader from gtfs2vec paper
- GTFS2Vec Model from gtfs2vec paper
- GTFS2Vec Embedder using gtfs2vec model
- Hex2Vec Model from hex2vec paper
- Hex2Vec Embedder using hex2vec model
- Highway2Vec Model from highway2vec paper
- Highway2Vec Embedder using highway2vec model
- OSMOnlineLoader
- OSMPbfLoader
- OSMWayLoader
- Neighbourhood
- H3Neighbourhood
- AdjacencyNeighbourhood
- CountEmbedder
- ContextualCountEmbedder
- (CI) Changelog Enforcer
- Utility plotting module based on Folium and Plotly
- Project README
- Documentation for srai library
- Citation information

### Changed

- Change embedders and joiners interface to have `.transform` method
- Change linter to Ruff and removed flake8, isort, pydocstyle
- Change default value inside `transform` function of IntersectionJoiner to not return geometry.
- Make torch and pytorch-lightning as optional dependencies ([#210](https://github.com/srai-lab/srai/issues/210))

### Deprecated

### Removed

### Fixed

- IntersectionJoiner incorrectly returned feature columns when `return_geom=False` ([#208](https://github.com/srai-lab/srai/issues/208))
- Tests for pandas >=2

### Security

## [0.0.1] - 2022-11-23

### Added

- PDM as a dependency management tool
- black, flake8, isort, mypy, pytest-cov
- pre-commit configuration
- Apache 2.0 license
- mkdocs for documentation
- GitHub pages to host documentation
- initial requirements
- H3 Regionizer
- Voronoi Regionizer
- Administrative Boundary Regionizer
- Intersection Joiner
- Geoparquet Loader

[unreleased]: https://github.com/srai-lab/srai/compare/0.1.1...HEAD
[0.1.1]: https://github.com/srai-lab/srai/compare/0.0.1...0.1.1
[0.0.1]: https://github.com/srai-lab/srai/compare/687500b...0.0.1
