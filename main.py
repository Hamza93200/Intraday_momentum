def daily_trading_routine():
    print("Starting daily trading routine at 22:00...")

    # Step 1: Close all open positions
    print("\n🔹 Closing all positions...")
    close_positions()

    # Step 2: Fetch historical price data
    print("\n🔹 Fetching historical price data for token pool...")
    df_prices = get_prices_df(token_pool)

    if df_prices is None or df_prices.empty:
        print("Error: No price data retrieved. Exiting routine.")
        return
    
    # Step 3: Compute daily returns and identify top & bottom tokens
    print("\n🔹 Computing daily returns and selecting best & worst tokens...")
    token_selection = get_tops(df_prices)

    print("\n✅ Selected Tokens:")
    print(token_selection)

    best_tokens = token_selection["Best Tokens"].dropna().tolist()
    worst_tokens = token_selection["Worst Tokens"].dropna().tolist()

    # Step 4: Get available USDT balance
    print("\n🔹 Checking available USDT balance...")
    usdt_balance = get_usdt_balance()

    if usdt_balance is None or float(usdt_balance) <= 100:
        print("Error: Insufficient USDT balance. Exiting routine.")
        return

    usdt_balance = float(usdt_balance)
    
    # Step 5: Calculate position sizes
    total_trades = len(best_tokens) + len(worst_tokens)
    if total_trades == 0:
        print("Error: No tokens selected for trading. Exiting routine.")
        return

    allocation_per_trade = (usdt_balance / total_trades)*1
    print(f"\n🔹 Allocating ${allocation_per_trade:.2f} per trade with 2x leverage.")

    # Step 6: Place orders
    print("\n🔹 Placing long positions on top 5 tokens...")
    for token in best_tokens:
        token_pair = token.replace("USDT", "") + "USDT"
        order_size = round(allocation_per_trade / df_prices[token_pair].iloc[-1], 8)  # Calculate order size
        print(f"📈 Placing LONG order: {token_pair} | Size: {order_size}")
        place_order(symbol=token_pair, size=order_size, side="buy",trade_side="open", leverage=1)

    print("\n🔹 Placing short positions on worst 5 tokens...")
    for token in worst_tokens:
        token_pair = token.replace("USDT", "") + "USDT"
        order_size = round(allocation_per_trade / df_prices[token_pair].iloc[-1], 6)
        print(f"📉 Placing SHORT order: {token_pair} | Size: {order_size}")
        place_order(symbol=token_pair, size=order_size, side="sell",trade_side="open", leverage=1)

    print("\n✅ Daily trading routine completed!")

daily_trading_routine()
