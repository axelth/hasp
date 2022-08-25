# Data analysis
- Document here the project: HASP
- Description: Audio Based Hazard Alert System for Pedestrieans
- Data Source: https://urbansounddataset.weebly.com/urbansound8k.html
- Type of analysis: Sound event classification and detection


# Install

## prerequisites
1. download dataset
2. install `https://github.com/axelth/us8kdata`
3. convert raw data using `us8kdata-convert`
    `us8kdata-convert /path/to/UrbanSound8K /path/to/clean-data-dir`

## Clone the project and install it:
 !Use the same virtual environment as for us8kdata-convert!
```bash
git clone git@github.com:axelth/hasp.git
cd hasp
```
and install the project in either development mode

```bash
pip install -e '.[dev]'
```

or as a regular package.

```bash
pip install .
```
