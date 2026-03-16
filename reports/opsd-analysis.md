# Open Power System Data - Renewable Energy Analysis

## Summary

Analysis of renewable energy installations across Belgium (BE), France (FR), and Germany (DE) using Open Power System Data to determine optimal weather data collection points.

## Data Coverage

### FR
- **Total plants**: 56,097
- **Plants with coordinates**: 55,856 (99.6%)
- **Energy types**:
  - Solar: 51,875 plants
  - Hydro: 2,115 plants
  - Wind: 1,592 plants
  - Bioenergy: 512 plants
  - Marine: 2 plants
- **Total capacity with coordinates**: 25524.2 MW

### DE
- **Total plants**: 1,461,180
- **Plants with coordinates**: 1,460,571 (100.0%)
- **Energy types**:
  - Solar: 1,426,692 plants
  - Wind: 14,864 plants
  - Bioenergy: 13,174 plants
  - Hydro: 6,443 plants
  - Geothermal: 7 plants
- **Total capacity with coordinates**: 67954.1 MW

## Weather Zone Configuration

Based on capacity-weighted K-means clustering of actual renewable energy installations:

### FR
#### Wind Onshore
- **Total capacity**: 14847.1 MW across 4 zones
- **Zone details**:
  - Zone 1: 44.097°N, 3.172°E, 11.8% weight, 1756.5 MW
  - Zone 2: 49.684°N, 2.310°E, 35.0% weight, 5189.7 MW
  - Zone 3: 47.476°N, -1.325°E, 21.7% weight, 3218.0 MW
  - Zone 4: 48.483°N, 4.546°E, 31.5% weight, 4683.0 MW

#### Solar
- **Total capacity**: 6008.9 MW across 4 zones
- **Zone details**:
  - Zone 1: 44.141°N, 0.810°E, 40.4% weight, 2430.5 MW
  - Zone 2: 48.385°N, 4.970°E, 11.1% weight, 668.4 MW
  - Zone 3: 44.025°N, 5.327°E, 31.8% weight, 1912.0 MW
  - Zone 4: 47.223°N, -0.306°E, 16.6% weight, 998.0 MW

#### Hydro
- **Total capacity**: 1996.8 MW across 4 zones
- **Zone details**:
  - Zone 1: 45.165°N, 5.813°E, 41.1% weight, 821.6 MW
  - Zone 2: 43.494°N, 1.308°E, 37.8% weight, 754.8 MW
  - Zone 3: 48.117°N, 5.965°E, 11.8% weight, 236.0 MW
  - Zone 4: 46.810°N, 1.114°E, 9.2% weight, 184.4 MW

#### Biomass
- **Total capacity**: 768.6 MW across 4 zones
- **Zone details**:
  - Zone 1: 47.695°N, -0.955°E, 16.8% weight, 129.1 MW
  - Zone 2: 47.655°N, 5.884°E, 14.2% weight, 109.3 MW
  - Zone 3: 44.478°N, 3.416°E, 30.9% weight, 237.1 MW
  - Zone 4: 49.373°N, 2.453°E, 38.1% weight, 293.1 MW

#### Wind Offshore
- **Total capacity**: 1800.0 MW across 2 zones
- **Zone details**:
  - Zone 1: 49.500°N, 1.000°E, 60.0% weight, 1000.0 MW
  - Zone 2: 47.500°N, -3.000°E, 40.0% weight, 800.0 MW

#### Load
- **Total capacity**: 0.0 MW across 5 zones
- **Zone details**:
  - Zone 1 (Paris): 48.857°N, 2.352°E, 20.0% weight, 0.0 MW
  - Zone 2 (Lyon): 45.764°N, 4.836°E, 8.0% weight, 0.0 MW
  - Zone 3 (Marseille): 43.297°N, 5.370°E, 5.0% weight, 0.0 MW
  - Zone 4 (Bordeaux): 44.838°N, -0.579°E, 4.0% weight, 0.0 MW
  - Zone 5 (Nantes): 47.218°N, -1.554°E, 3.0% weight, 0.0 MW

### DE
#### Wind Onshore
- **Total capacity**: 23282.0 MW across 4 zones
- **Zone details**:
  - Zone 1: 53.325°N, 8.656°E, 31.5% weight, 7342.7 MW
  - Zone 2: 50.387°N, 7.992°E, 24.2% weight, 5631.1 MW
  - Zone 3: 51.537°N, 12.154°E, 26.0% weight, 6064.1 MW
  - Zone 4: 53.376°N, 12.740°E, 18.2% weight, 4244.2 MW

#### Solar
- **Total capacity**: 15861.3 MW across 4 zones
- **Zone details**:
  - Zone 1: 52.131°N, 12.762°E, 40.9% weight, 6489.0 MW
  - Zone 2: 52.630°N, 8.306°E, 19.0% weight, 3009.3 MW
  - Zone 3: 49.417°N, 8.511°E, 16.6% weight, 2631.8 MW
  - Zone 4: 48.896°N, 11.415°E, 23.5% weight, 3731.2 MW

#### Hydro
- **Total capacity**: 1286.3 MW across 4 zones
- **Zone details**:
  - Zone 1: 48.644°N, 8.917°E, 35.9% weight, 461.9 MW
  - Zone 2: 48.341°N, 11.909°E, 26.0% weight, 334.2 MW
  - Zone 3: 51.405°N, 8.778°E, 17.1% weight, 220.6 MW
  - Zone 4: 50.959°N, 12.938°E, 21.0% weight, 269.6 MW

#### Biomass
- **Total capacity**: 7590.0 MW across 4 zones
- **Zone details**:
  - Zone 1: 53.251°N, 9.826°E, 22.0% weight, 1670.7 MW
  - Zone 2: 52.258°N, 12.745°E, 19.5% weight, 1480.6 MW
  - Zone 3: 48.852°N, 10.720°E, 34.0% weight, 2580.9 MW
  - Zone 4: 51.739°N, 7.797°E, 24.5% weight, 1857.9 MW

#### Wind Offshore
- **Total capacity**: 7500.0 MW across 3 zones
- **Zone details**:
  - Zone 1: 54.500°N, 6.500°E, 40.0% weight, 3000.0 MW
  - Zone 2: 54.700°N, 7.500°E, 40.0% weight, 3000.0 MW
  - Zone 3: 54.800°N, 14.000°E, 20.0% weight, 1500.0 MW

#### Load
- **Total capacity**: 0.0 MW across 5 zones
- **Zone details**:
  - Zone 1 (Berlin): 52.520°N, 13.405°E, 15.0% weight, 0.0 MW
  - Zone 2 (Munich): 48.135°N, 11.582°E, 8.0% weight, 0.0 MW
  - Zone 3 (Hamburg): 53.551°N, 9.994°E, 8.0% weight, 0.0 MW
  - Zone 4 (Cologne): 50.938°N, 6.960°E, 6.0% weight, 0.0 MW
  - Zone 5 (Frankfurt): 50.111°N, 8.682°E, 4.0% weight, 0.0 MW

### BE
#### Wind Onshore
- **Total capacity**: 2000.0 MW across 4 zones
- **Zone details**:
  - Zone 1 (West Flanders): 50.800°N, 3.500°E, 30.0% weight, 600.0 MW
  - Zone 2 (Antwerp Province): 51.100°N, 4.200°E, 40.0% weight, 800.0 MW
  - Zone 3 (Brabant): 50.500°N, 5.000°E, 20.0% weight, 400.0 MW
  - Zone 4 (Eastern Belgium): 50.300°N, 5.800°E, 10.0% weight, 200.0 MW

#### Solar
- **Total capacity**: 3000.0 MW across 4 zones
- **Zone details**:
  - Zone 1 (Central Belgium): 50.800°N, 4.300°E, 40.0% weight, 1200.0 MW
  - Zone 2 (Northern Belgium): 51.100°N, 4.800°E, 30.0% weight, 900.0 MW
  - Zone 3 (Southern Belgium): 50.400°N, 4.000°E, 20.0% weight, 600.0 MW
  - Zone 4 (Eastern Belgium): 50.200°N, 5.500°E, 10.0% weight, 300.0 MW

#### Wind Offshore
- **Total capacity**: 2000.0 MW across 2 zones
- **Zone details**:
  - Zone 1 (North Sea Zone 1): 51.600°N, 2.800°E, 50.0% weight, 1000.0 MW
  - Zone 2 (North Sea Zone 2): 51.400°N, 2.900°E, 50.0% weight, 1000.0 MW

#### Hydro
- **Total capacity**: 300.0 MW across 2 zones
- **Zone details**:
  - Zone 1 (Ardennes): 50.200°N, 5.800°E, 60.0% weight, 200.0 MW
  - Zone 2 (Central Valleys): 50.400°N, 4.500°E, 40.0% weight, 100.0 MW

#### Load
- **Total capacity**: 0.0 MW across 4 zones
- **Zone details**:
  - Zone 1 (Brussels): 50.850°N, 4.352°E, 35.0% weight, 0.0 MW
  - Zone 2 (Antwerp): 51.219°N, 4.402°E, 20.0% weight, 0.0 MW
  - Zone 3 (Ghent): 51.054°N, 3.717°E, 15.0% weight, 0.0 MW
  - Zone 4 (Liège): 50.633°N, 5.580°E, 10.0% weight, 0.0 MW

## Methodology

1. **Data Source**: Open Power System Data (OPSD) renewable power plants dataset
2. **Countries**: Analysis of FR and DE from OPSD data; BE zones created manually
3. **Clustering**: Capacity-weighted K-means with 4 clusters per energy type
4. **Offshore Wind**: Manual placement based on known wind farm locations
5. **Load Zones**: Population-weighted major city locations

## Notes

- **Belgium**: Limited data availability in OPSD, zones created based on geographic knowledge
- **Offshore Wind**: Manual coordinates used as offshore installations often lack precise coordinates
- **Load Forecasting**: Major population centers used as proxy for electricity demand
- **Weighting**: Based on installed electrical capacity (MW) for generation zones
