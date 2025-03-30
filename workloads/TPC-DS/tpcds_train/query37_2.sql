--start query 1 IN stream 1 using template query37.tpl
SELECT i.i_item_id AS item_id,
       i.i_item_desc AS item_desc,
       i.i_current_price AS current_price
FROM item AS i, inventory AS inv, date_dim AS d, catalog_sales AS cs
WHERE i.i_current_price BETWEEN 28 AND 28 + 30
  AND inv.inv_item_sk = i.i_item_sk
  AND d.d_date_sk = inv.inv_date_sk
  AND d.d_date BETWEEN CAST('1998-01-16' AS DATE) AND CAST('1998-03-17' AS DATE)
  AND i.i_manufact_id IN (831, 791, 815, 826)
  AND inv.inv_quantity_on_hand BETWEEN 100 AND 500
  AND cs.cs_item_sk = i.i_item_sk
GROUP BY i.i_item_id, i.i_item_desc, i.i_current_price
ORDER BY i.i_item_id
LIMIT 100;
