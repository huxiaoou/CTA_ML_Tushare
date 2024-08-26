$bgn_date = "20120104"
$stp_date = "20240826"

# ------------------------
# --- remove existence ---
# ------------------------
Remove-Item -Recurse d:\OneDrive\Data\Projects\CTA_ML_Tushare\*

# ----------------------------
# --- exectue all projects ---
# ----------------------------

python main.py --bgn $bgn_date --stp $stp_date available
python main.py --bgn $bgn_date --stp $stp_date market
python main.py --bgn $bgn_date --stp $stp_date test_return

python main.py --bgn $bgn_date --stp $stp_date factor --fclass MTM
python main.py --bgn $bgn_date --stp $stp_date factor --fclass SKEW
python main.py --bgn $bgn_date --stp $stp_date factor --fclass RS
