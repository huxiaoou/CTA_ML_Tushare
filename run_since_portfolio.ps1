$bgn_date = "20120104"
$bgn_date_ml = "20170201" # machine learning bgn date
$bgn_date_sig = "20170703" # signal bgn date
$bgn_date_sim = "20180102" # simulation bgn date
$stp_date = "20241008"

# -----------------------
# --- remove old data ---
# -----------------------

Write-Host "Removing data ..."

Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare\signals\portfolios\*
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare\simulations\portfolios\*
Remove-Item -Recurse E:\Data\Projects\CTA_ML_Tushare\evaluations\portfolios\*

# ----------------------------
# --- exectue all projects ---
# ----------------------------

python main.py --bgn $bgn_date_sig --stp $stp_date signals --type portfolios
python main.py --bgn $bgn_date_sim --stp $stp_date simulations --type portfolios
python main.py --bgn $bgn_date_sim --stp $stp_date simulations --type omega
python main.py --bgn $bgn_date_sim --stp $stp_date evaluations --type portfolios
