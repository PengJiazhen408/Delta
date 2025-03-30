--start query 1 IN stream 0 using template query91.tpl
SELECT
  cc.cc_call_center_id AS Call_Center,
  cc.cc_name AS Call_Center_Name,
  cc.cc_manager AS Manager,
  SUM(cr.cr_net_loss) AS Returns_Loss
FROM
  call_center AS cc
  ,catalog_returns AS cr
  ,date_dim AS d
  ,customer AS c
  ,customer_address AS ca
  ,customer_demographics AS cd
  ,household_demographics AS hd
WHERE
  cr.cr_call_center_sk = cc.cc_call_center_sk
  AND cr.cr_returned_date_sk = d.d_date_sk
  AND cr.cr_returning_customer_sk = c.c_customer_sk
  AND cd.cd_demo_sk = c.c_current_cdemo_sk
  AND hd.hd_demo_sk = c.c_current_hdemo_sk
  AND ca.ca_address_sk = c.c_current_addr_sk
  AND d.d_year = 1999
  AND d.d_moy = 11
  AND (
    (cd.cd_marital_status = 'M' AND cd.cd_education_status = 'Unknown')
    OR (cd.cd_marital_status = 'W' AND cd.cd_education_status = 'Advanced Degree')
  )
  AND hd.hd_buy_potential LIKE '>10000%'
  AND ca.ca_gmt_offset = -7
GROUP BY
  cc.cc_call_center_id,
  cc.cc_name,
  cc.cc_manager,
  cd.cd_marital_status,
  cd.cd_education_status
ORDER BY
  SUM(cr.cr_net_loss) DESC;