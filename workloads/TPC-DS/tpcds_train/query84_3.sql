--start query 1 IN stream 0 using template query84.tpl
SELECT
  c.c_customer_id AS customer_id,
  COALESCE(c.c_last_name, '') || ', ' || COALESCE(c.c_first_name, '') AS customername
FROM
  customer AS c
  ,customer_address AS ca
  ,customer_demographics AS cd
  ,household_demographics AS hd
  ,income_band AS ib
  ,store_returns AS sr
WHERE
  ca.ca_city = 'Pleasant Hill'
  AND c.c_current_addr_sk = ca.ca_address_sk
  AND ib.ib_lower_bound >= 35577
  AND ib.ib_upper_bound <= 35577 + 50000
  AND ib.ib_income_band_sk = hd.hd_income_band_sk
  AND cd.cd_demo_sk = c.c_current_cdemo_sk
  AND hd.hd_demo_sk = c.c_current_hdemo_sk
  AND sr.sr_cdemo_sk = cd.cd_demo_sk
ORDER BY
  c.c_customer_id
LIMIT 100;