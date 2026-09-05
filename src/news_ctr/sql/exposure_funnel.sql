WITH counts AS (
    SELECT
        COUNT(DISTINCT user_id)::BIGINT AS active_users,
        COUNT(DISTINCT impression_id)::BIGINT AS impressions,
        COUNT(*)::BIGINT AS candidates,
        SUM(label)::BIGINT AS clicks
    FROM candidates
), funnel AS (
    SELECT 1 AS stage_order, 'active_users' AS stage, active_users AS units,
           1.0::DOUBLE AS rate_from_previous
    FROM counts
    UNION ALL
    SELECT 2, 'impressions', impressions, impressions::DOUBLE / NULLIF(active_users, 0)
    FROM counts
    UNION ALL
    SELECT 3, 'candidates', candidates, candidates::DOUBLE / NULLIF(impressions, 0)
    FROM counts
    UNION ALL
    SELECT 4, 'clicks', clicks, clicks::DOUBLE / NULLIF(candidates, 0)
    FROM counts
)
SELECT stage, units, rate_from_previous
FROM funnel
ORDER BY stage_order;
