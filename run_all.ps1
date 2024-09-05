$bgn_date = "20120104"
$bgn_date_ml = "20170201" # machine learning bgn date
$bgn_date_sig = "20170703" # signal bgn date
$bgn_date_sim = "20180102" # simulation bgn date
$stp_date = "20240826"

# ------------------------
# --- remove existence ---
# ------------------------
Remove-Item -Recurse E:\OneDrive\Data\Projects\CTA_ML_Tushare\*

# ----------------------------
# --- exectue all projects ---
# ----------------------------

# --- prepare
python main.py --bgn $bgn_date --stp $stp_date available
python main.py --bgn $bgn_date --stp $stp_date market
python main.py --bgn $bgn_date --stp $stp_date test_return --type calculate
python main.py --bgn $bgn_date --stp $stp_date test_return --type regroup

# --- factor
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

# --- machine learning
python main.py --bgn $bgn_date_ml --stp $stp_date --processes 16 feature_selection
python main.py --bgn $bgn_date_ml --stp $stp_date mclrn --type parse
python main.py --bgn $bgn_date_ml --stp $stp_date --processes 16 mclrn --type trnprd

# --- model signals and simulation ---
python main.py --bgn $bgn_date_sig --stp $stp_date signals --type models
python main.py --bgn $bgn_date_sim --stp $stp_date simulations --type models
python main.py --bgn $bgn_date_sim --stp $stp_date evaluations --type models

python main.py --bgn $bgn_date_sig --stp $stp_date signals --type portfolios
python main.py --bgn $bgn_date_sim --stp $stp_date simulations --type portfolios
python main.py --bgn $bgn_date_sim --stp $stp_date evaluations --type portfolios
