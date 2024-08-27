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
python main.py --bgn $bgn_date --stp $stp_date factor --fclass BASIS
python main.py --bgn $bgn_date --stp $stp_date factor --fclass TS
python main.py --bgn $bgn_date --stp $stp_date factor --fclass S0BETA
python main.py --bgn $bgn_date --stp $stp_date factor --fclass S1BETA
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CBETA
python main.py --bgn $bgn_date --stp $stp_date factor --fclass IBETA
python main.py --bgn $bgn_date --stp $stp_date factor --fclass PBETA
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CTP
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CTR
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CVP
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CVR
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CSP
python main.py --bgn $bgn_date --stp $stp_date factor --fclass CSR
python main.py --bgn $bgn_date --stp $stp_date factor --fclass NOI
python main.py --bgn $bgn_date --stp $stp_date factor --fclass NDOI
python main.py --bgn $bgn_date --stp $stp_date factor --fclass WNOI
python main.py --bgn $bgn_date --stp $stp_date factor --fclass WNDOI
