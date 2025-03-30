--start query 1 IN stream 2 using template query20.tpl
SELECT i.i_item_id AS item_id,
       i.i_item_desc AS item_desc,
       i.i_category AS category,
       i.i_class AS class,
       i.i_current_price AS current_price,
       SUM(cs.cs_ext_sales_price) AS itemrevenue,
       SUM(cs.cs_ext_sales_price) * 100 / SUM(SUM(cs.cs_ext_sales_price)) OVER (PARTITION BY i.i_class) AS revenueratio
FROM catalog_sales AS cs, item AS i, date_dim AS d
WHERE cs.cs_item_sk = i.i_item_sk
  AND i.i_category IN ('Books', 'Shoes', 'Electronics')
  AND cs.cs_sold_date_sk = d.d_date_sk
  AND d.d_date BETWEEN CAST('1998-03-21' AS DATE) AND CAST('1998-04-20' AS DATE)
GROUP BY i.i_item_id, i.i_item_desc, i.i_category, i.i_class, i.i_current_price
ORDER BY i.i_category, i.i_class, i.i_item_id, i.i_item_desc, revenueratio
LIMIT 100;
