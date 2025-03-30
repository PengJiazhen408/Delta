--start query 1 IN stream 1 using template query82.tpl
SELECT i.i_item_id,
       i.i_item_desc,
       i.i_current_price
FROM item AS i, inventory AS inv, date_dim AS d, store_sales AS ss
WHERE i.i_current_price BETWEEN 3 AND 3 + 30
      AND inv.inv_item_sk = i.i_item_sk
      AND d.d_date_sk = inv.inv_date_sk
      AND d.d_date BETWEEN CAST('1998-05-20' AS DATE) AND CAST('1998-07-19' AS DATE)
      AND i.i_manufact_id IN (59,526,301,399)
      AND inv.inv_quantity_on_hand BETWEEN 100 AND 500
      AND ss.ss_item_sk = i.i_item_sk
GROUP BY i.i_item_id, i.i_item_desc, i.i_current_price
ORDER BY i.i_item_id
-- LIMIT 100;
