# Run with the old error function by default
python error_calculation.py

# Run with the new error function (element-wise difference, norm (divided by original), then mean)
python error_calculation.py --use_new_error