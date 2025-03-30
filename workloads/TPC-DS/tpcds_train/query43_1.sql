--start query 1 IN stream 0 using template query43.tpl
SELECT s.s_store_name,
       s.s_store_id,
       SUM(CASE WHEN (d.d_day_name = 'Sunday') THEN ss.ss_sales_price ELSE NULL END) AS sun_sales,
       SUM(CASE WHEN (d.d_day_name = 'Monday') THEN ss.ss_sales_price ELSE NULL END) AS mon_sales,
       SUM(CASE WHEN (d.d_day_name = 'Tuesday') THEN ss.ss_sales_price ELSE NULL END) AS tue_sales,
       SUM(CASE WHEN (d.d_day_name = 'Wednesday') THEN ss.ss_sales_price ELSE NULL END) AS wed_sales,
       SUM(CASE WHEN (d.d_day_name = 'Thursday') THEN ss.ss_sales_price ELSE NULL END) AS thu_sales,
       SUM(CASE WHEN (d.d_day_name = 'Friday') THEN ss.ss_sales_price ELSE NULL END) AS fri_sales,
       SUM(CASE WHEN (d.d_day_name = 'Saturday') THEN ss.ss_sales_price ELSE NULL END) AS sat_sales
FROM date_dim AS d, store_sales AS ss, store AS s
WHERE d.d_date_sk = ss.ss_sold_date_sk
  AND s.s_store_sk = ss.ss_store_sk
  AND s.s_gmt_offset = -5
  AND d.d_year = 1998
GROUP BY s.s_store_name, s.s_store_id
ORDER BY s.s_store_name, s.s_store_id, sun_sales, mon_sales, tue_sales, wed_sales, thu_sales, fri_sales, sat_sales
LIMIT 100;
