$bgn_date = "20120104"
$bgn_date_ml = "20170201" # machine learning bgn date
$bgn_date_sig = "20170703" # signal bgn date
$bgn_date_sim = "20180102" # simulation bgn date
$stp_date = "20240826"

# ------------------------
# --- remove existence ---
# ------------------------
Remove-Item -Recurse E:\OneDrive\Data\Projects\CTA_ML_Tushare\signals\portfolios\*
Remove-Item -Recurse E:\OneDrive\Data\Projects\CTA_ML_Tushare\simulations\portfolios\*
Remove-Item -Recurse E:\OneDrive\Data\Projects\CTA_ML_Tushare\evaluations\portfolios\*

# ----------------------------
# --- exectue all projects ---
# ----------------------------

python main.py --bgn $bgn_date_sig --stp $stp_date signals --type portfolios
python main.py --bgn $bgn_date_sim --stp $stp_date simulations --type portfolios
python main.py --bgn $bgn_date_sim --stp $stp_date evaluations --type portfolios
