
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <ctime>
#include <optional>
#include <memory>
#include <numeric>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <stdexcept>

namespace execution {

// ---------------------------------------------------------------------------
// Utility – minimal log_message stand-in
// ---------------------------------------------------------------------------
inline void log_message(const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    std::cout << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << " | " << msg << "\n";
}

// ---------------------------------------------------------------------------
// Helper – current time as time_point
// ---------------------------------------------------------------------------
using Clock     = std::chrono::system_clock;
using TimePoint = Clock::time_point;

inline std::string time_point_to_iso(const TimePoint& tp) {
    auto t = Clock::to_time_t(tp);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

inline int hour_of(const TimePoint& tp) {
    auto t = Clock::to_time_t(tp);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    return tm_buf.tm_hour;
}

inline TimePoint replace_hour(const TimePoint& tp, int hour) {
    auto t = Clock::to_time_t(tp);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    tm_buf.tm_hour = hour;
    tm_buf.tm_min  = 0;
    tm_buf.tm_sec  = 0;
    return Clock::from_time_t(std::mktime(&tm_buf));
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------
enum class OrderType {
    MARKET,
    LIMIT,
    STOP,
    STOP_LIMIT,
    TWAP,        // Time-Weighted Average Price
    VWAP,        // Volume-Weighted Average Price
    PARTICIPATE   // Participate with % of volume
};

inline std::string to_string(OrderType t) {
    switch (t) {
        case OrderType::MARKET:      return "MARKET";
        case OrderType::LIMIT:       return "LIMIT";
        case OrderType::STOP:        return "STOP";
        case OrderType::STOP_LIMIT:  return "STOP_LIMIT";
        case OrderType::TWAP:        return "TWAP";
        case OrderType::VWAP:        return "VWAP";
        case OrderType::PARTICIPATE: return "PARTICIPATE";
    }
    return "UNKNOWN";
}

enum class OrderSide { BUY, SELL };

inline std::string to_string(OrderSide s) {
    return s == OrderSide::BUY ? "BUY" : "SELL";
}

enum class OrderStatus {
    PENDING,
    SUBMITTED,
    PARTIALLY_FILLED,
    FILLED,
    CANCELLED,
    REJECTED,
    ERROR
};

inline std::string to_string(OrderStatus s) {
    switch (s) {
        case OrderStatus::PENDING:          return "PENDING";
        case OrderStatus::SUBMITTED:        return "SUBMITTED";
        case OrderStatus::PARTIALLY_FILLED: return "PARTIALLY_FILLED";
        case OrderStatus::FILLED:           return "FILLED";
        case OrderStatus::CANCELLED:        return "CANCELLED";
        case OrderStatus::REJECTED:         return "REJECTED";
        case OrderStatus::ERROR:            return "ERROR";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// Market-data snapshot for a single ticker
// ---------------------------------------------------------------------------
struct MarketDataEntry {
    double    price     = 0.0;
    double    volume    = 0.0;
    TimePoint timestamp = Clock::now();
};

// ---------------------------------------------------------------------------
// Fill record
// ---------------------------------------------------------------------------
struct Fill {
    int       quantity  = 0;
    double    price     = 0.0;
    TimePoint timestamp = Clock::now();
};

// ---------------------------------------------------------------------------
// Order
// ---------------------------------------------------------------------------
class Order {
public:
    Order(const std::string& ticker,
          OrderSide side,
          int quantity,
          OrderType order_type                    = OrderType::MARKET,
          std::optional<double> limit_price       = std::nullopt,
          std::optional<double> stop_price        = std::nullopt,
          const std::string& time_in_force        = "DAY",
          std::optional<double> participation_rate = std::nullopt,
          std::optional<TimePoint> start_time     = std::nullopt,
          std::optional<TimePoint> end_time       = std::nullopt)
        : ticker_(ticker)
        , side_(side)
        , quantity_(quantity)
        , order_type_(order_type)
        , limit_price_(limit_price)
        , stop_price_(stop_price)
        , time_in_force_(time_in_force)
        , participation_rate_(participation_rate)
        , start_time_(start_time.value_or(Clock::now()))
        , end_time_(end_time)
        , status_(OrderStatus::PENDING)
        , filled_quantity_(0)
        , average_fill_price_(0.0)
        , commission_(0.0)
    {
        // Build a deterministic-ish order id
        auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
                         Clock::now().time_since_epoch()).count();
        auto hash_val = std::hash<std::string>{}(ticker) % 10000;
        order_id_ = "ORD-" + std::to_string(epoch) + "-" + std::to_string(hash_val);
    }

    // -- Status management ---------------------------------------------------
    void update_status(OrderStatus new_status, const std::string& message = "") {
        OrderStatus old_status = status_;
        status_ = new_status;

        if (new_status == OrderStatus::SUBMITTED)  submitted_time_  = Clock::now();
        if (new_status == OrderStatus::FILLED)     filled_time_     = Clock::now();
        if (new_status == OrderStatus::CANCELLED)  cancelled_time_  = Clock::now();
        if (new_status == OrderStatus::REJECTED)   rejection_reason_ = message;

        std::string log_msg = "Order " + order_id_ + " (" + ticker_ + ") status changed: "
                            + to_string(old_status) + " -> " + to_string(new_status);
        if (!message.empty()) log_msg += " - " + message;
        log_message(log_msg);
    }

    // -- Fill management -----------------------------------------------------
    void add_fill(int quantity, double price, std::optional<TimePoint> timestamp = std::nullopt) {
        Fill fill;
        fill.quantity  = quantity;
        fill.price     = price;
        fill.timestamp = timestamp.value_or(Clock::now());
        fills_.push_back(fill);

        double old_value = filled_quantity_ * average_fill_price_;
        double new_value = static_cast<double>(quantity) * price;
        filled_quantity_ += quantity;

        if (filled_quantity_ > 0) {
            average_fill_price_ = (old_value + new_value) / filled_quantity_;
        }

        if (filled_quantity_ >= quantity_) {
            update_status(OrderStatus::FILLED);
        } else if (filled_quantity_ > 0) {
            update_status(OrderStatus::PARTIALLY_FILLED);
        }
    }

    // -- Cancel --------------------------------------------------------------
    bool cancel(const std::string& reason = "") {
        if (status_ == OrderStatus::PENDING ||
            status_ == OrderStatus::SUBMITTED ||
            status_ == OrderStatus::PARTIALLY_FILLED)
        {
            update_status(OrderStatus::CANCELLED, reason);
            return true;
        }
        return false;
    }

    // -- Accessors -----------------------------------------------------------
    const std::string& order_id()        const { return order_id_; }
    const std::string& ticker()          const { return ticker_; }
    OrderSide          side()            const { return side_; }
    int                quantity()         const { return quantity_; }
    OrderType          order_type()      const { return order_type_; }
    OrderStatus        status()          const { return status_; }
    int                filled_quantity()  const { return filled_quantity_; }
    double             average_fill_price() const { return average_fill_price_; }
    double             commission()       const { return commission_; }
    TimePoint          start_time()      const { return start_time_; }
    std::optional<TimePoint> end_time()  const { return end_time_; }
    const std::vector<Fill>&                fills()        const { return fills_; }
    const std::vector<std::shared_ptr<Order>>& child_orders() const { return child_orders_; }

    // Mutable references for algorithms to manipulate
    std::vector<std::shared_ptr<Order>>& child_orders() { return child_orders_; }
    void set_order_type(OrderType t)       { order_type_ = t; }
    void set_quantity(int q)               { quantity_ = q; }
    void set_start_time(TimePoint tp)      { start_time_ = tp; }
    void set_filled_quantity(int q)        { filled_quantity_ = q; }
    void set_average_fill_price(double p)  { average_fill_price_ = p; }
    void set_broker_order_id(const std::string& id) { broker_order_id_ = id; }

private:
    std::string  order_id_;
    std::string  ticker_;
    OrderSide    side_;
    int          quantity_;
    OrderType    order_type_;
    std::optional<double> limit_price_;
    std::optional<double> stop_price_;
    std::string  time_in_force_;
    std::optional<double> participation_rate_;
    TimePoint    start_time_;
    std::optional<TimePoint> end_time_;

    OrderStatus  status_;
    int          filled_quantity_;
    double       average_fill_price_;
    std::vector<Fill> fills_;
    double       commission_;

    std::optional<TimePoint>  submitted_time_;
    std::optional<TimePoint>  filled_time_;
    std::optional<TimePoint>  cancelled_time_;
    std::string               rejection_reason_;
    std::string               broker_order_id_;

    std::vector<std::shared_ptr<Order>> child_orders_;
};

// ---------------------------------------------------------------------------
// Execution Algorithm – base class
// ---------------------------------------------------------------------------
class ExecutionAlgorithm {
public:
    explicit ExecutionAlgorithm(const std::string& name) : name_(name) {}
    virtual ~ExecutionAlgorithm() = default;

    virtual std::vector<std::shared_ptr<Order>>
    create_execution_plan(Order& order) = 0;

    /// Returns true when execution of the parent order is complete.
    virtual bool execute_step(Order& order,
                              const std::unordered_map<std::string, MarketDataEntry>& market_data) = 0;

    const std::string& name() const { return name_; }

protected:
    std::string name_;
};

// ---------------------------------------------------------------------------
// TWAP Algorithm
// ---------------------------------------------------------------------------
class TWAPAlgorithm : public ExecutionAlgorithm {
public:
    explicit TWAPAlgorithm(int num_slices = 10, int slice_interval_minutes = 5)
        : ExecutionAlgorithm("TWAP")
        , num_slices_(num_slices)
        , slice_interval_minutes_(slice_interval_minutes)
    {}

    std::vector<std::shared_ptr<Order>>
    create_execution_plan(Order& order) override {
        if (order.order_type() != OrderType::TWAP) {
            order.set_order_type(OrderType::TWAP);
        }

        int slice_size = std::max(1, order.quantity() / num_slices_);
        int remainder  = order.quantity() % num_slices_;

        std::vector<std::shared_ptr<Order>> child_orders;
        TimePoint start = order.start_time();

        for (int i = 0; i < num_slices_; ++i) {
            int qty = slice_size + (i == num_slices_ - 1 ? remainder : 0);
            if (qty <= 0) continue;

            auto child = std::make_shared<Order>(
                order.ticker(),
                order.side(),
                qty,
                OrderType::MARKET,
                std::nullopt, std::nullopt, "DAY");

            child->set_start_time(start + std::chrono::minutes(i * slice_interval_minutes_));
            child_orders.push_back(child);
        }

        order.child_orders() = child_orders;
        return child_orders;
    }

    bool execute_step(Order& order,
                      const std::unordered_map<std::string, MarketDataEntry>& market_data) override
    {
        auto now = Clock::now();

        for (auto& child : order.child_orders()) {
            if (child->status() == OrderStatus::PENDING && child->start_time() <= now) {
                child->update_status(OrderStatus::SUBMITTED);

                auto it = market_data.find(child->ticker());
                if (it != market_data.end() && it->second.price > 0.0) {
                    child->add_fill(child->quantity(), it->second.price);
                }
            }
        }

        bool all_filled = std::all_of(
            order.child_orders().begin(), order.child_orders().end(),
            [](const std::shared_ptr<Order>& c) { return c->status() == OrderStatus::FILLED; });

        if (all_filled) {
            int    total_qty   = 0;
            double total_value = 0.0;
            for (auto& c : order.child_orders()) {
                total_qty   += c->filled_quantity();
                total_value += c->filled_quantity() * c->average_fill_price();
            }
            order.set_filled_quantity(total_qty);
            order.set_average_fill_price(total_qty > 0 ? total_value / total_qty : 0.0);

            if (order.filled_quantity() >= order.quantity()) {
                order.update_status(OrderStatus::FILLED);
            } else if (order.filled_quantity() > 0) {
                order.update_status(OrderStatus::PARTIALLY_FILLED);
            }
            return true;
        }
        return false;
    }

private:
    int num_slices_;
    int slice_interval_minutes_;
};

// ---------------------------------------------------------------------------
// VWAP Algorithm
// ---------------------------------------------------------------------------
class VWAPAlgorithm : public ExecutionAlgorithm {
public:
    explicit VWAPAlgorithm(
        const std::unordered_map<int, double>& volume_profile = {})
        : ExecutionAlgorithm("VWAP")
    {
        if (volume_profile.empty()) {
            volume_profile_ = {
                { 9, 0.12}, {10, 0.09}, {11, 0.08}, {12, 0.07},
                {13, 0.08}, {14, 0.09}, {15, 0.22}, {16, 0.25}
            };
        } else {
            volume_profile_ = volume_profile;
        }
    }

    std::vector<std::shared_ptr<Order>>
    create_execution_plan(Order& order) override {
        if (order.order_type() != OrderType::VWAP) {
            order.set_order_type(OrderType::VWAP);
        }

        int start_hour = hour_of(order.start_time());
        int end_hour   = order.end_time().has_value() ? hour_of(*order.end_time()) : 16;

        // Compute total volume weight in the window
        double total_volume = 0.0;
        for (int h = start_hour; h <= end_hour; ++h) {
            auto it = volume_profile_.find(h);
            if (it != volume_profile_.end()) total_volume += it->second;
        }

        // Build a normalised profile for the window
        std::unordered_map<int, double> profile;
        if (total_volume == 0.0) {
            int hours = std::max(1, end_hour - start_hour + 1);
            for (int h = start_hour; h <= end_hour; ++h)
                profile[h] = 1.0 / hours;
        } else {
            for (int h = start_hour; h <= end_hour; ++h) {
                auto it = volume_profile_.find(h);
                profile[h] = (it != volume_profile_.end()) ? it->second / total_volume : 0.0;
            }
        }

        // Collect and sort hours
        std::vector<int> hours;
        hours.reserve(profile.size());
        for (auto& [h, _] : profile) hours.push_back(h);
        std::sort(hours.begin(), hours.end());

        std::vector<std::shared_ptr<Order>> child_orders;
        int remaining = order.quantity();

        for (int h : hours) {
            if (h < start_hour || h > end_hour) continue;

            int qty = static_cast<int>(order.quantity() * profile[h]);
            qty = std::min(qty, remaining);
            if (qty <= 0) continue;

            auto child = std::make_shared<Order>(
                order.ticker(), order.side(), qty,
                OrderType::MARKET,
                std::nullopt, std::nullopt, "DAY");

            TimePoint scheduled = replace_hour(order.start_time(), h);
            if (scheduled < order.start_time()) scheduled = order.start_time();
            child->set_start_time(scheduled);

            child_orders.push_back(child);
            remaining -= qty;
        }

        // Add leftover to the last child
        if (remaining > 0 && !child_orders.empty()) {
            auto& last = child_orders.back();
            last->set_quantity(last->quantity() + remaining);
        }

        order.child_orders() = child_orders;
        return child_orders;
    }

    bool execute_step(Order& order,
                      const std::unordered_map<std::string, MarketDataEntry>& market_data) override
    {
        auto now = Clock::now();

        for (auto& child : order.child_orders()) {
            if (child->status() == OrderStatus::PENDING && child->start_time() <= now) {
                child->update_status(OrderStatus::SUBMITTED);

                auto it = market_data.find(child->ticker());
                if (it != market_data.end() && it->second.price > 0.0) {
                    child->add_fill(child->quantity(), it->second.price);
                }
            }
        }

        bool all_filled = std::all_of(
            order.child_orders().begin(), order.child_orders().end(),
            [](const std::shared_ptr<Order>& c) { return c->status() == OrderStatus::FILLED; });

        if (all_filled) {
            int    total_qty   = 0;
            double total_value = 0.0;
            for (auto& c : order.child_orders()) {
                total_qty   += c->filled_quantity();
                total_value += c->filled_quantity() * c->average_fill_price();
            }
            order.set_filled_quantity(total_qty);
            order.set_average_fill_price(total_qty > 0 ? total_value / total_qty : 0.0);

            if (order.filled_quantity() >= order.quantity()) {
                order.update_status(OrderStatus::FILLED);
            } else if (order.filled_quantity() > 0) {
                order.update_status(OrderStatus::PARTIALLY_FILLED);
            }
            return true;
        }
        return false;
    }

private:
    std::unordered_map<int, double> volume_profile_;
};

// ---------------------------------------------------------------------------
// Execution Manager
// ---------------------------------------------------------------------------
class ExecutionManager {
public:
    ExecutionManager() {
        // Ensure cache directory exists (mirrors the Python os.makedirs call)
        std::filesystem::create_directories("cache/execution");

        algorithms_[OrderType::TWAP] = std::make_unique<TWAPAlgorithm>();
        algorithms_[OrderType::VWAP] = std::make_unique<VWAPAlgorithm>();
    }

    // -- Order creation ------------------------------------------------------
    std::shared_ptr<Order> create_order(
        const std::string& ticker,
        OrderSide side,
        int quantity,
        OrderType order_type                        = OrderType::MARKET,
        std::optional<double> limit_price           = std::nullopt,
        std::optional<double> stop_price            = std::nullopt,
        const std::string& time_in_force            = "DAY",
        std::optional<double> participation_rate    = std::nullopt,
        std::optional<TimePoint> start_time         = std::nullopt,
        std::optional<TimePoint> end_time           = std::nullopt)
    {
        auto order = std::make_shared<Order>(
            ticker, side, quantity, order_type,
            limit_price, stop_price, time_in_force,
            participation_rate, start_time, end_time);

        orders_[order->order_id()] = order;
        return order;
    }

    // -- Order submission ----------------------------------------------------
    bool submit_order(std::shared_ptr<Order> order) {
        if (orders_.find(order->order_id()) == orders_.end()) {
            orders_[order->order_id()] = order;
        }

        // Algorithmic order?
        if (algorithms_.count(order->order_type())) {
            algorithms_[order->order_type()]->create_execution_plan(*order);
            order->update_status(OrderStatus::SUBMITTED);
            return true;
        }

        // Simple order – submit and simulate immediate fill
        order->update_status(OrderStatus::SUBMITTED);

        double price = 100.0; // default
        auto it = market_data_.find(order->ticker());
        if (it != market_data_.end()) price = it->second.price;

        order->add_fill(order->quantity(), price);
        return true;
    }

    // -- Order cancellation --------------------------------------------------
    bool cancel_order(const std::string& order_id) {
        auto it = orders_.find(order_id);
        if (it == orders_.end()) return false;
        return it->second->cancel("Cancelled by user");
    }

    // -- Market data ---------------------------------------------------------
    void update_market_data(const std::string& ticker, double price, double volume = 0.0) {
        market_data_[ticker] = MarketDataEntry{price, volume, Clock::now()};
    }

    // -- Processing loop -----------------------------------------------------
    int process_orders() {
        int processed = 0;

        for (auto& [order_id, order] : orders_) {
            if (order->status() == OrderStatus::SUBMITTED ||
                order->status() == OrderStatus::PARTIALLY_FILLED)
            {
                auto alg_it = algorithms_.find(order->order_type());
                if (alg_it != algorithms_.end()) {
                    bool complete = alg_it->second->execute_step(*order, market_data_);
                    ++processed;
                    if (complete) {
                        log_message("Order " + order_id + " execution completed");
                    }
                }
            }
        }
        return processed;
    }

    // -- Accessors -----------------------------------------------------------
    std::shared_ptr<Order> get_order(const std::string& id) const {
        auto it = orders_.find(id);
        return it != orders_.end() ? it->second : nullptr;
    }

private:
    std::unordered_map<std::string, std::shared_ptr<Order>>          orders_;
    std::unordered_map<OrderType, std::unique_ptr<ExecutionAlgorithm>> algorithms_;
    std::unordered_map<std::string, MarketDataEntry>                  market_data_;
};

} // namespace execution

// ---------------------------------------------------------------------------
// Demo / test main
// ---------------------------------------------------------------------------
int main() {
    using namespace execution;

    std::cout << "=== TradeSmart Execution Engine Demo ===\n\n";

    ExecutionManager mgr;

    // Seed market data
    mgr.update_market_data("BHP.AX",  45.20, 1'200'000);
    mgr.update_market_data("CBA.AX", 110.50,   800'000);
    mgr.update_market_data("WBC.AX",  27.80,   950'000);

    // --- 1. Simple MARKET order ---
    std::cout << "--- MARKET order ---\n";
    auto mkt = mgr.create_order("BHP.AX", OrderSide::BUY, 500, OrderType::MARKET);
    mgr.submit_order(mkt);
    std::cout << "  Filled qty   : " << mkt->filled_quantity()    << "\n";
    std::cout << "  Avg fill px  : " << mkt->average_fill_price() << "\n\n";

    // --- 2. TWAP order (10 slices, 1-min apart so they fire immediately) ---
    std::cout << "--- TWAP order (10 slices) ---\n";
    auto twap = mgr.create_order("CBA.AX", OrderSide::SELL, 1000, OrderType::TWAP);
    mgr.submit_order(twap);
    // Drive the processing loop until done or 20 iterations
    for (int i = 0; i < 20 && twap->status() != OrderStatus::FILLED; ++i)
        mgr.process_orders();
    std::cout << "  Status       : " << to_string(twap->status())         << "\n";
    std::cout << "  Filled qty   : " << twap->filled_quantity()           << "\n";
    std::cout << "  Avg fill px  : " << twap->average_fill_price()        << "\n";
    std::cout << "  Child orders : " << twap->child_orders().size()       << "\n\n";

    // --- 3. VWAP order ---
    std::cout << "--- VWAP order ---\n";
    auto vwap = mgr.create_order("WBC.AX", OrderSide::BUY, 2000, OrderType::VWAP);
    mgr.submit_order(vwap);
    for (int i = 0; i < 20 && vwap->status() != OrderStatus::FILLED; ++i)
        mgr.process_orders();
    std::cout << "  Status       : " << to_string(vwap->status())         << "\n";
    std::cout << "  Filled qty   : " << vwap->filled_quantity()           << "\n";
    std::cout << "  Avg fill px  : " << vwap->average_fill_price()        << "\n";
    std::cout << "  Child orders : " << vwap->child_orders().size()       << "\n\n";

    // --- 4. Cancel a pending order ---
    std::cout << "--- Cancel test ---\n";
    auto limit = mgr.create_order("BHP.AX", OrderSide::BUY, 100, OrderType::LIMIT,
                                  /*limit_price=*/44.00);
    limit->update_status(OrderStatus::SUBMITTED);
    bool cancelled = mgr.cancel_order(limit->order_id());
    std::cout << "  Cancel result: " << (cancelled ? "OK" : "FAILED") << "\n";
    std::cout << "  Status       : " << to_string(limit->status())    << "\n\n";

    std::cout << "=== Demo complete ===\n";
    return 0;
}

} // namespace execution