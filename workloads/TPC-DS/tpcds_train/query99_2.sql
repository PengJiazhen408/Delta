--start query 1 IN stream 0 using template query99.tpl
SELECT
  SUBSTRING(w.w_warehouse_name, 1, 20) AS warehouse_name,
  sm.sm_type,
  cc.cc_name,
  SUM(CASE WHEN (cs.cs_ship_date_sk - cs.cs_sold_date_sk <= 30) THEN 1 ELSE 0 END) AS "30 days",
  SUM(CASE WHEN (cs.cs_ship_date_sk - cs.cs_sold_date_sk > 30) AND
               (cs.cs_ship_date_sk - cs.cs_sold_date_sk <= 60) THEN 1 ELSE 0 END) AS "31-60 days",
  SUM(CASE WHEN (cs.cs_ship_date_sk - cs.cs_sold_date_sk > 60) AND
               (cs.cs_ship_date_sk - cs.cs_sold_date_sk <= 90) THEN 1 ELSE 0 END) AS "61-90 days",
  SUM(CASE WHEN (cs.cs_ship_date_sk - cs.cs_sold_date_sk > 90) AND
               (cs.cs_ship_date_sk - cs.cs_sold_date_sk <= 120) THEN 1 ELSE 0 END) AS "91-120 days",
  SUM(CASE WHEN (cs.cs_ship_date_sk - cs.cs_sold_date_sk > 120) THEN 1 ELSE 0 END) AS ">120 days"
FROM
  catalog_sales AS cs
  ,warehouse AS w
  ,ship_mode AS sm
  ,call_center AS cc
  ,date_dim AS d
WHERE
  d.d_month_seq BETWEEN 1190 AND 1190 + 11
  AND cs.cs_ship_date_sk = d.d_date_sk
  AND cs.cs_warehouse_sk = w.w_warehouse_sk
  AND cs.cs_ship_mode_sk = sm.sm_ship_mode_sk
  AND cs.cs_call_center_sk = cc.cc_call_center_sk
GROUP BY
  SUBSTRING(w.w_warehouse_name, 1, 20),
  sm.sm_type,
  cc.cc_name
ORDER BY
  SUBSTRING(w.w_warehouse_name, 1, 20),
  sm.sm_type,
  cc.cc_name
LIMIT 100;
