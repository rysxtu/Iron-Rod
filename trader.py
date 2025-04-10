import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias
import math
import statistics

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

# id token: eyJraWQiOiJ4M3NhZjFZTkNsRGwyVDljemdCR01ybnVVMlJlNDNjb1E1UGxYMWgwb2tBPSIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiJjZjcyZDllZC0wOGRkLTQ4ZjEtOTFhZS0wYmFjNmVkMjAxODMiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaXNzIjoiaHR0cHM6XC9cL2NvZ25pdG8taWRwLmV1LXdlc3QtMS5hbWF6b25hd3MuY29tXC9ldS13ZXN0LTFfek9mVngwcWl3IiwiY29nbml0bzp1c2VybmFtZSI6ImNmNzJkOWVkLTA4ZGQtNDhmMS05MWFlLTBiYWM2ZWQyMDE4MyIsIm9yaWdpbl9qdGkiOiIxYTIxZjNkOC1mZGE1LTQ0YmMtODQ5Mi01MjU0ZWRjYjc2NTUiLCJhdWQiOiIzMmM1ZGM1dDFrbDUxZWRjcXYzOWkwcjJzMiIsImV2ZW50X2lkIjoiNDdlZTRkNjgtMjRhMS00YTI0LTg0ZDktOTBkNzMyNDhjOTBlIiwidG9rZW5fdXNlIjoiaWQiLCJhdXRoX3RpbWUiOjE3NDE4NTM3MDQsImV4cCI6MTc0NDAyOTE1MywiaWF0IjoxNzQ0MDI1NTUzLCJqdGkiOiJlNzg5MzIxYS03MTY5LTQ1OWEtYjY2ZC00MzViZmU2YzlmODAiLCJlbWFpbCI6ImFsdTAxMjMyM0BnbWFpbC5jb20ifQ.hQ-Ov_lNvu0i82TMWlAIMplBOS2J9pohP4n05p7DXXbFJUFKtJjshQO0hDkKkOESnujl2cTypF38jjhX7HsEJkH5nKzHPEosRY7_DI4UJ1pOn_RvlSp_1CUunt5QjbjTCI-u6tCrlld0ieFF5pxsXorDe4HrAeB3_Im-7Xcj5IAiI6L_Y_OesjcYhZ9hDBALhQ5Cgn56OAqduqGxCj1KDkOegtlXCbPGNaPQAZcFLZgDmQnwQTaNHfN_c38A_GydnKH1HlSOeMk5WIo6ytgS84k2cXHr7AyDYplNtr52pqet6qPnZ7EUrTjYm9PHJs4iPoem4mM6BadjntQLUTa7nQ

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

logger = Logger()

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    # the actual method that is implemented by concrete strategies
    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    # loads the orders based on the state and how a concrete inherited model acts
    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    # append buy orders, which are positive quantities
    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    # append sell orders, which are negative quantities
    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    # save to json? not doing anything
    def save(self) -> JSON:
        return None

    # idk some other non-method
    def load(self, data: JSON) -> None:
        pass

class MarketMakingStrategy(Strategy):

    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        # the length of data series they are looking at?
        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        # get order book for the given product
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        # get the current state
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # not too sure? check if the "window" is filled out
        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        # slowly sell out vs quickly?
        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        # willing to buy/sell more expensive/cheaper if closer to the limit
        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value
        
        # just finds the right amount to buy
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        # some sort of hedging? doesn't buy out all the positions to the limit at once - only does so if soft_liquidate
        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        # more buying??
        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        # does the exact same for buy orders, in the case for selling
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data)

class ResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        # idk how this strat is earning so much
        return 10_000

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol] # this is just getting the current order book for kelp
        # len(order_depth) would be the number of trades being put on
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True) # sorted to get most expensive buy price for asking
        sell_orders = sorted(order_depth.sell_orders.items())             # sorted to get the least expensive sell price for bidding

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0] # higher buy price for asking
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0] # least sell price for buying

        return round((popular_buy_price + popular_sell_price) / 2) # their "fair value" is the average of the hi/lo 

class SquidStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        # swing a bit, but some say there is a pattern to be discovered in its prize progression
        return 2000
    # Sum of prices / Number of Observations

class SquidStrategy2(MarketMakingStrategy):
    #true value = current mean
    def get_true_value(self, state: TradingState) -> int:
        # swing a bit, but some say there is a pattern to be discovered in its prize progression
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys(), default=1970)
        best_ask = min(order_depth.sell_orders.keys(), default=1970)
        return (best_bid + best_ask) // 2
class SquidStrategy3(MarketMakingStrategy):
    #mean reverting
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.price_history = deque(maxlen=100)  # Track last 100 prices
        self.mean_price = None
        self.std_dev = None

    def get_true_value(self, state: TradingState) -> int:
        # Update price history with current mid price
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        
        if best_bid and best_ask:
            current_price = (best_bid + best_ask) / 2
            self.price_history.append(current_price)
            
            # Calculate mean and standard deviation when we have enough data
            if len(self.price_history) >= 20:
                prices = list(self.price_history)
                self.mean_price = sum(prices) / len(prices)
                self.std_dev = (sum((x - self.mean_price) ** 2 for x in prices) / len(prices)) ** 0.5
                
                # Mean-reverting logic
                if current_price > self.mean_price + self.std_dev:
                    # Price is high - adjust true value downward
                    return int(self.mean_price) + int(self.std_dev/2)
                elif current_price < self.mean_price - self.std_dev:
                    # Price is low - adjust true value upward
                    return int(self.mean_price) - int(self.std_dev/2)
        
        # Default to simple mid price if not enough data
        return int((best_bid + best_ask) / 2) if best_bid and best_ask else 1900
class SquidStrategy4(MarketMakingStrategy):
    #true value = mean of last 10 values
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.price_history = deque(maxlen=10)  # Track last 10 prices

    def get_true_value(self, state: TradingState) -> int:
        # swing a bit, but some say there is a pattern to be discovered in its prize progression
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys(), default=2000)
        best_ask = min(order_depth.sell_orders.keys(), default=2000)
        self.price_history.append(int((best_bid + best_ask) // 2))
        if len(self.price_history) >= 5:
            return int(sum(list(self.price_history)) // len(list(self.price_history)))
        return ((best_bid + best_ask) // 2)
class SquidStrategy5(MarketMakingStrategy):
    #strategy assuming kelp moves oppositely to squid ink
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.paired_symbol = "KELP"
        self.hedge_ratio = 2
        self.intercept = 6000
        self.kelp_price_history = deque(maxlen=15)

    def get_true_value(self, state: TradingState) -> int:
        kelp_order_depth = state.order_depths.get(self.paired_symbol)
        squid_order_depth = state.order_depths.get(self.symbol)
        best_bid = max(squid_order_depth.buy_orders.keys(), default=2000)
        best_ask = min(squid_order_depth.sell_orders.keys(), default=2000)
        if not kelp_order_depth or not squid_order_depth:
            return (best_bid + best_ask)//2

        kelp_best_bid = max(kelp_order_depth.buy_orders.keys(), default=1000)
        kelp_best_ask = min(kelp_order_depth.sell_orders.keys(), default=1000)
        kelp_mid = (kelp_best_bid + kelp_best_ask) // 2

        self.kelp_price_history.append(kelp_mid)

        if len(self.kelp_price_history) < 15:
            return (best_bid + best_ask)//2

        # Calculate trend (delta)
        kelp_trend = self.kelp_price_history[-1] + self.kelp_price_history[-2] - self.kelp_price_history[-14] - self.kelp_price_history[-13]

        # Inverse trend adjustment
        estimated_squid_value = (best_bid + best_ask)//2 - 2 * kelp_trend

        return int(estimated_squid_value)
class SquidStrategy6(MarketMakingStrategy):
    #momentum strategy
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.price_history = deque(maxlen=100)
        self.momentum_window = 101  # Number of steps to measure momentum
        self.amplitude = 50
        self.period = 6 * 60 * 1000  # 6 minutes in ms (tune based on sim time)
        self.phase_shift = math.pi

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return 1830  # fallback

        best_bid = max(order_depth.buy_orders.keys(), default=1830)
        best_ask = min(order_depth.sell_orders.keys(), default=1830)
        mid_price = (best_bid + best_ask) // 2
        self.price_history.append(mid_price)



        # --- Sine Wave for Time of Day Pattern ---
        time_ms = state.timestamp
        sine_adjustment = self.amplitude * math.sin((2 * math.pi / self.period) * time_ms + self.phase_shift)
        
        # --- Momentum ---
        if len(self.price_history) >= self.momentum_window:
            momentum = self.price_history[-1] - self.price_history[-self.momentum_window]
        elif len(self.price_history) == self.momentum_window-1:
            momentum = 0
        else:
            momentum = self.price_history[-1] - self.price_history[-len(self.price_history)] - 50

        

        # --- Combine ---
        true_value = mid_price - 1.4 * momentum + 1.1 * sine_adjustment #best is -1.4 momentum, 1.1 sine
        return int(true_value)
class SquidStrategy7(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.price_history = deque(maxlen=100)
        self.error_history = deque(maxlen=5)
        self.default_price = 1830

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return self.default_price

        best_bid = max(order_depth.buy_orders.keys(), default=self.default_price)
        best_ask = min(order_depth.sell_orders.keys(), default=self.default_price)
        mid_price = (best_bid + best_ask) // 2
        self.price_history.append(mid_price)

        if len(self.price_history) < 100:
            return mid_price

        diff_series = [self.price_history[i] - self.price_history[i - 1] for i in range(1, len(self.price_history))]

        predicted_change = diff_series[-1]

        avg_error = sum(self.error_history) / len(self.error_history) if self.error_history else 0

        prediction = self.price_history[-1] + predicted_change + avg_error

        # Track error
        error = mid_price - prediction
        self.error_history.append(error)

        return int(prediction)
class SquidStrategy8(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.symbol = symbol
        self.limit = limit
        self.imbalance_history = deque(maxlen=20)
        self.imbalance_threshold = 0.6  # Ratio threshold (e.g., 60% of volume on one side)

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths.get(self.symbol)
        if not order_depth:
            return 1830  # fallback value

        best_bid = max(order_depth.buy_orders.keys(), default=1830)
        best_ask = min(order_depth.sell_orders.keys(), default=1830)
        mid_price = (best_bid + best_ask) // 2

        # Total bid and ask volume in the book
        total_bid_volume = sum(order_depth.buy_orders.values())
        total_ask_volume = sum(order_depth.sell_orders.values())

        # Prevent division by zero
        total_volume = total_bid_volume + total_ask_volume
        if total_volume == 0:
            return mid_price

        # Order book imbalance as a percentage
        imbalance = (total_bid_volume - total_ask_volume) / total_volume
        self.imbalance_history.append(imbalance)

        # Detect recent spike in imbalance
        recent_imbalance = imbalance
        avg_imbalance = sum(self.imbalance_history) / len(self.imbalance_history)

        # Adjustment factor based on spike direction
        if abs(recent_imbalance - avg_imbalance) > self.imbalance_threshold:
            adjustment = 10 * (1 if recent_imbalance > 0 else -1)
        else:
            adjustment = 0

        return int(mid_price + adjustment)
class SquidStrategy9(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.price_history = deque(maxlen=300)

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)

        if best_bid is None or best_ask is None:
            if len(self.price_history):
                return int(self.price_history[-1])
            return 1880# fallback if no valid prices

        mid_price = (best_bid + best_ask) / 2
        self.price_history.append(mid_price)

        if len(self.price_history) < 20:
            return int(mid_price)

        # Calculate mean and standard deviation
        prices = list(self.price_history)
        mean_price = sum(prices) / len(prices)
        std_dev = (sum((x - mean_price) ** 2 for x in prices) / len(prices)) ** 0.5

        if std_dev == 0:
            return int(mid_price)  # avoid division by zero

        # Calculate z-score (how far current price is from mean)
        z_score = (mid_price - mean_price) / std_dev

        # Adjust true value *toward* the mean, with a factor based on z-score
        adjustment_strength = 0.6  # how strongly we revert to the mean
        true_value = mid_price - adjustment_strength * z_score * std_dev
        

        return int(true_value)


class SquidStrategy10(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.price_history = deque(maxlen=100)
        self.mean_price = None
        self.std_dev = None

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        
        if best_bid is None or best_ask is None:
            return 1850  # fallback

        current_price = (best_bid + best_ask) / 2
        self.price_history.append(current_price)

        if len(self.price_history) < 20:
            return int(current_price)

        # --- Mean and Standard Deviation ---
        prices = list(self.price_history)
        self.mean_price = sum(prices) / len(prices)
        self.std_dev = (sum((x - self.mean_price) ** 2 for x in prices) / len(prices)) ** 0.5

        # --- Momentum Filter ---
        recent_momentum = self.price_history[-1] - self.price_history[-5]
        
        # --- Spike Detection and Reversion Entry ---
        if current_price > self.mean_price + 2 * self.std_dev:
            # strong upward spike — only revert if price starts dropping
            if self.price_history[-1] < self.price_history[-2] and recent_momentum < 0:
                return int(self.mean_price) + int(self.std_dev)
            else:
                return int(current_price)  # wait for reversal confirmation

        elif current_price < self.mean_price - 2 * self.std_dev:
            # strong downward spike — only revert if price starts rising
            if self.price_history[-1] > self.price_history[-2] and recent_momentum > 0:
                return int(self.mean_price) - int(self.std_dev)
            else:
                return int(current_price)  # wait for reversal confirmation

        # Small deviations — normal mean reversion
        if current_price > self.mean_price + self.std_dev:
            return int(self.mean_price) + int(self.std_dev / 2)
        elif current_price < self.mean_price - self.std_dev:
            return int(self.mean_price) - int(self.std_dev / 2)

        return int(current_price)
class SquidStrategy11(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.price_history = deque(maxlen=75)
        self.mean_price = None
        self.std_dev = None
        self.adjustment_strength = 0.4  # Tune this value

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            if (len(self.price_history)):
                return self.price_history[-1]
            return 1950  # fallback

        current_price = (best_bid + best_ask) / 2
        self.price_history.append(current_price)

        if len(self.price_history) < 20:
            return int(current_price)

        prices = list(self.price_history)
        self.mean_price = sum(prices) / len(prices)
        self.std_dev = (sum((x - self.mean_price) ** 2 for x in prices) / len(prices)) ** 0.5
        #maybe change this to more local mean
        if current_price > self.mean_price + 2 * self.std_dev:
            return int(self.mean_price + self.std_dev)  # Sell only
        elif current_price < self.mean_price - 2 * self.std_dev:
            return int(self.mean_price - self.std_dev)  # Buy only

        # Deviation from mean → use it to adjust the true value
        deviation = current_price - self.mean_price
        k = -self.adjustment_strength * deviation
        true_value = current_price + k

        return int(true_value)


class Trader:
    def __init__(self) -> None:
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }

        self.strategies = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": ResinStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK": SquidStrategy11, 

        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        #logger.print(state.position)

        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol, None))

            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
    
