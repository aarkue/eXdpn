## p2p_base: 
- 3 decision points
- 1: Request Manager Approval if total_price >= 800, Request Standard Approval if total_price <= 1000
- 2: Manager Rejection if total_price >= 500 and random_exponential(100, 200) > 120, no guard for Manager Approval 
- 3: Standard Rejection if total_price >= 500 and random_exponential(100, 200) > 120, no guard for Standard Approval 

## p2p_base_no_randomness: 
- 3 decision points
- 1: Request Manager Approval if total_price >= 800, otherwise Request Standard Approval 
- 2: Manager Rejection if "unprofessional" items are being bought, i.e., "RECARO Exo Gaming Stuhl" or "Ducky One 3 Matcha TKL". Otherwise, Approval
- 3: Standard Rejection if item_category is "Fun" or if the Supplier is "Scamming Corp." 

## p2p_nonlinearities: 
- 5 decision points
- 1: Request Manager Approval if total_price >= 600 or supplier "Dunder Mifflin"; Request Standard Approval if total_price <= 1200
- 2: Manager Reject Purchase if total_price >= 500 and amount mod 2 = 1 (no guard for Manager Approval)
- 3: Standard Reject Purchase if total_price >= 500 and amount mod 2 = 1 (no guard for Standard Approval)
- 3: Goods damaged if item_amount^3 >= total_price; Goods Fine if item_amount^3 < total_price
- 4: Revocation Costumer if supplierMap(supplier) >= amount (see petri net figure for map function); Revocation Vendor if total_price/amount < 100; Cancel Payment if Receive Invoice and Goods Damaged and Pay Invoice if Receive Invoice and Goods Fine
- 5: Pay Invoice if Goods Fine and Cancel Order if Revocation Costumer/Vendor 
	
## p2p_with_time_based_discount:
- 4 decision points
- 1: Request Manager Approval if total_price >= 800, Request Standard Approval if total_price <= 1000
- 2: Manager Rejection if total_price >= 500 and random_exponential(100, 200) > 120, no guard for Manager Approval 
- 3: Standard Rejection if total_price >= 500 and random_exponential(100, 200) > 120, no guard for Standard Approval 
- 4: Send Order to Vendor if the delay since the previous event is smaller than some threshold; Send Order to Vendor with Discount if the delay since the previous event is larger than some threshold. This threshold depends on wether or not the order was approved by a manager (i.e., the previous event name).

