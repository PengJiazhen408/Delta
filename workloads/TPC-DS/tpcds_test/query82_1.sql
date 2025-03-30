--start query 1 IN stream 0 using template query82.tpl
SELECT i.i_item_id,
       i.i_item_desc,
       i.i_current_price
FROM item AS i, inventory AS inv, date_dim AS d, store_sales AS ss
WHERE i.i_current_price BETWEEN 30 AND 30 + 30
      AND inv.inv_item_sk = i.i_item_sk
      AND d.d_date_sk = inv.inv_date_sk
      AND d.d_date BETWEEN CAST('2002-05-30' AS DATE) AND CAST('2002-07-29' AS DATE)
      AND i.i_manufact_id IN (437,129,727,663)
      AND inv.inv_quantity_on_hand BETWEEN 100 AND 500
      AND ss.ss_item_sk = i.i_item_sk
GROUP BY i.i_item_id, i.i_item_desc, i.i_current_price
ORDER BY i.i_item_id
-- LIMIT 100;
