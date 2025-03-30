--start query 1 IN stream 0 using template query62.tpl
SELECT
  SUBSTRING(w.w_warehouse_name, 1, 20) AS warehouse_name,
  sm.sm_type,
  web_site.web_name,
  SUM(CASE WHEN (ws.ws_ship_date_sk - ws.ws_sold_date_sk <= 30) THEN 1 ELSE 0 END) AS "30 days",
  SUM(CASE WHEN (ws.ws_ship_date_sk - ws.ws_sold_date_sk > 30) AND
               (ws.ws_ship_date_sk - ws.ws_sold_date_sk <= 60) THEN 1 ELSE 0 END) AS "31-60 days",
  SUM(CASE WHEN (ws.ws_ship_date_sk - ws.ws_sold_date_sk > 60) AND
               (ws.ws_ship_date_sk - ws.ws_sold_date_sk <= 90) THEN 1 ELSE 0 END) AS "61-90 days",
  SUM(CASE WHEN (ws.ws_ship_date_sk - ws.ws_sold_date_sk > 90) AND
               (ws.ws_ship_date_sk - ws.ws_sold_date_sk <= 120) THEN 1 ELSE 0 END) AS "91-120 days",
  SUM(CASE WHEN (ws.ws_ship_date_sk - ws.ws_sold_date_sk > 120) THEN 1 ELSE 0 END) AS ">120 days"
FROM
  web_sales AS ws
  ,warehouse AS w
  ,ship_mode AS sm
  ,web_site AS web_site 
  ,date_dim AS d
WHERE
  d.d_month_seq BETWEEN 1190 AND 1190 + 11
  AND ws.ws_ship_date_sk = d.d_date_sk
  AND ws.ws_warehouse_sk = w.w_warehouse_sk
  AND ws.ws_ship_mode_sk = sm.sm_ship_mode_sk
  AND ws.ws_web_site_sk = web_site.web_site_sk
GROUP BY
  SUBSTRING(w.w_warehouse_name, 1, 20),
  sm.sm_type,
  web_site.web_name
ORDER BY
  SUBSTRING(w.w_warehouse_name, 1, 20),
  sm.sm_type,
  web_site.web_name
LIMIT 100;
