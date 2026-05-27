# How to Export Data from Football Manager

Place your FM export files in this folder.

## Step-by-step export guide

### Option A — League Table (easiest)
1. In FM go to your league → **Standings / Table** view
2. Make sure these columns are visible: Team, MP, W, D, L, GF, GA, Pts
3. Press **Ctrl+P** → choose **"Web Page (HTML)"** → save here as e.g. `league_table.html`

### Option B — Squad Stats
1. Go to your club → **Squad** view
2. Right-click any column header and add: Apps, Goals, Clean Sheets
3. Press **Ctrl+P** → export as HTML → save here

### Option C — CSV (FM Touch / newer versions)
Some FM versions let you export directly as CSV.
Choose that option and save the file here as e.g. `squad_stats.csv`

## After exporting
Run:
```
python src/fm_import.py
```

This will list all teams found and show an example prediction.

To predict a specific match in the Streamlit app, go to the
**Football Manager** tab and select your teams from the dropdown.
