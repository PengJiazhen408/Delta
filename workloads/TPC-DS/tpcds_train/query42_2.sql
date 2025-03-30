--start query 1 IN stream 1 using template query42.tpl
SELECT dt.d_year,
       i.i_category_id AS category_id,
       i.i_category AS category,
       SUM(ss.ss_ext_sales_price) AS total_ext_sales_price
FROM date_dim AS dt, store_sales AS ss, item AS i
WHERE dt.d_date_sk = ss.ss_sold_date_sk
  AND ss.ss_item_sk = i.i_item_sk
  AND i.i_manager_id = 1
  AND dt.d_moy = 11
  AND dt.d_year = 2001
GROUP BY dt.d_year, i.i_category_id, i.i_category
ORDER BY total_ext_sales_price DESC, dt.d_year, i.i_category_id, i.i_category
LIMIT 100;
