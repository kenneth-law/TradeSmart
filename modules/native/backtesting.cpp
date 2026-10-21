#include <pybind11/pybind11.h>
#include <cmath>
#include <utility>

namespace py = pybind11;

double estimate_spread(double price, double volume, double volatility) {
    double base_spread = price < 5 ? 0.03 : price < 20 ? 0.01 : 0.005;
    

    double volume_factor = 1.0;
    if (volume < 100000) volume_factor = 1.5;
    else if (volume < 500000) volume_factor = 1.2;
    else if (volume < 1000000) volume_factor = 1.1;

    double volatility_factor = 1.0 + (volatility / 15.0);
    return base_spread * volume_factor * volatility_factor;
}

double estimate_market_impact(double price, double volume, double trade_size) {
    if (volume <= 0) return 0.0;

    double trade_volume_pct = (trade_size / volume) * 100.0;
    if (trade_volume_pct < 0.1) return 0.0;

    double impact = 0.05 * std::sqrt(trade_volume_pct / 20.0);

    if (price > 100) impact *= 0.6;
    else if (price > 50) impact *= 0.8;

    return std::min(impact, 0.5);
}

py::tuple calculate_transaction_cost(
    double price,
    int shares,
    double volume,
    double volatility,
    double custom_cost,
    const std::string& custom_type,
    bool has_custom_cost
) {
    double total_cost = 0.0;
    double notional = price * shares;

    if (has_custom_cost) {
        if (custom_type == "fixed") {
            total_cost = custom_cost;
        } else if (custom_type == "percent") {
            total_cost = notional * (custom_cost / 100.0);
        } else {
            total_cost = custom_cost * shares;
        }
    } else {
        double spread_pct = estimate_spread(price, volume, volatility);
        double spread_cost = price * spread_pct / 2.0;

        double impact_pct = estimate_market_impact(price, volume, shares);
        double impact_cost = price * impact_pct;

        double commission = 10.0;
        total_cost = (spread_cost + impact_cost) * shares + commission;
    }

    double cost_pct = notional > 0 ? (total_cost / notional) * 100.0 : 0.0;
    return py::make_tuple(total_cost, cost_pct);
}

PYBIND11_MODULE(_backtesting_cpp, m) {
    m.doc() = "Native backtesting helpers for TradeSmart Analytics";

    m.def("estimate_spread", &estimate_spread);
    m.def("estimate_market_impact", &estimate_market_impact);
    m.def("calculate_transaction_cost", &calculate_transaction_cost);
}