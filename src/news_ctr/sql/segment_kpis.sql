WITH segmented AS (
    SELECT 'device_type' AS segment, COALESCE(CAST(device_type AS VARCHAR), 'unknown') AS value,
           impression_id, label
    FROM candidates
    UNION ALL
    SELECT 'candidate_position', CAST(candidate_position AS VARCHAR), impression_id, label
    FROM candidates
    UNION ALL
    SELECT 'candidate_count',
           CASE
               WHEN candidate_count <= 5 THEN '01-05'
               WHEN candidate_count <= 10 THEN '06-10'
               ELSE '11+'
           END,
           impression_id, label
    FROM candidates
    UNION ALL
    SELECT 'history_length',
           CASE
               WHEN history_length = 0 THEN '0'
               WHEN history_length <= 3 THEN '1-3'
               WHEN history_length <= 9 THEN '4-9'
               ELSE '10+'
           END,
           impression_id, label
    FROM candidates
    UNION ALL
    SELECT 'freshness',
           CASE
               WHEN article_age_days < 1 THEN '<1d'
               WHEN article_age_days < 7 THEN '1-7d'
               ELSE '7d+'
           END,
           impression_id, label
    FROM candidates
)
SELECT
    segment,
    value,
    COUNT(DISTINCT impression_id)::BIGINT AS units,
    COUNT(*)::BIGINT AS candidates,
    SUM(label)::BIGINT AS clicks,
    SUM(label)::DOUBLE / NULLIF(COUNT(*), 0) AS ctr
FROM segmented
GROUP BY segment, value
ORDER BY segment, value;
