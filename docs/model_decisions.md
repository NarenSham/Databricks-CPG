External signals tested: CPI, gas prices, Google Trends
Result: No MAPE improvement over lag-only model
Reason: lag_1m and lag_12m already capture 98%+ of variance
        External signals add noise at category/province grain
Decision: Revert to lag-only feature set (10 features)
          Revisit signals at SKU/brand level if data becomes available