--start query 1 IN stream 0 using template query3.tpl
SELECT dt.d_year,
       i.i_brand_id AS brand_id,
       i.i_brand AS brand,
       SUM(ss.ss_ext_sales_price) AS sum_agg
FROM date_dim AS dt
     ,store_sales AS ss
     ,item AS i
WHERE dt.d_date_sk = ss.ss_sold_date_sk
      AND ss.ss_item_sk = i.i_item_sk
      AND i.i_manufact_id = 436
      AND dt.d_moy = 12
GROUP BY dt.d_year,
         i.i_brand,
         i.i_brand_id
ORDER BY dt.d_year,
         sum_agg DESC,
         brand_id
LIMIT 100;
