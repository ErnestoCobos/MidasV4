"""
Configuration for Binance commission rates.

This module contains settings for Binance commission rates with consideration for
different VIP levels and payment options.
"""

# Standard fee rate for Binance VIP 0 (0.1%)
standard_fee_rate = 0.001

# Discounted fee rate if using BNB for fees (25% discount, 0.075%)
bnb_discount_rate = 0.00075  

# Whether to use BNB to pay fees (enabling this applies the discounted rate)
use_bnb_for_fees = False

# Special trading pairs with zero fees (empty by default)
pairs_with_zero_fee = []

# Future extension: VIP level tiers
# VIP level fee structure for reference:
# VIP 0: 0.1% (maker) / 0.1% (taker)
# VIP 1: 0.09% / 0.1%
# VIP 2: 0.08% / 0.1%
# VIP 3: 0.07% / 0.09%
# VIP 4: 0.06% / 0.08%
# VIP 5: 0.05% / 0.07%
# VIP 6: 0.04% / 0.06%
# VIP 7: 0.03% / 0.05%
# VIP 8: 0.02% / 0.04%
# VIP 9: 0.015% / 0.035%