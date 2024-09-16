$bgn_date = "20120104"
$bgn_date_ml = "20170201" # machine learning bgn date
$bgn_date_sig = "20170703" # signal bgn date
$bgn_date_sim = "20180102" # simulation bgn date
$stp_date = "20240902"

Write-Host "Removing data ..."

# ------------------------
# --- remove existence ---
# ------------------------
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare\feature_selection\*
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare\mclrn\*
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare\signals\*
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare\simulations\*
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare\evaluations\*


# ----------------------------
# --- exectue all projects ---
# ----------------------------

Write-Host "Execute since feature selection ..."

# --- feature selection
python main.py --bgn $bgn_date_ml --stp $stp_date --processes 12 feature_selection

# --- machine learning
python main.py --bgn $bgn_date_ml --stp $stp_date mclrn --type parse
python main.py --bgn $bgn_date_ml --stp $stp_date --processes 12 mclrn --type trnprd

# --- model signals and simulation ---
python main.py --bgn $bgn_date_sig --stp $stp_date signals --type models
python main.py --bgn $bgn_date_sim --stp $stp_date simulations --type models
python main.py --bgn $bgn_date_sim --stp $stp_date evaluations --type models

python main.py --bgn $bgn_date_sig --stp $stp_date signals --type portfolios
python main.py --bgn $bgn_date_sim --stp $stp_date simulations --type portfolios
python main.py --bgn $bgn_date_sim --stp $stp_date evaluations --type portfolios
