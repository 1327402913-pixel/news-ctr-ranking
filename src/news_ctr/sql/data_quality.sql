WITH quality AS (
    SELECT 'candidate_rows' AS metric, COUNT(*)::DOUBLE AS numeric_value,
           NULL::VARCHAR AS text_value
    FROM candidates
    UNION ALL
    SELECT 'impression_rows', COUNT(DISTINCT impression_id)::DOUBLE, NULL::VARCHAR
    FROM candidates
    UNION ALL
    SELECT 'unique_users', COUNT(DISTINCT user_id)::DOUBLE, NULL::VARCHAR
    FROM candidates
    UNION ALL
    SELECT 'duplicate_impression_article',
           (COUNT(*) - COUNT(DISTINCT (impression_id, article_id)))::DOUBLE, NULL::VARCHAR
    FROM candidates
    UNION ALL
    SELECT 'impression_candidate_count_mismatch', COUNT(*)::DOUBLE, NULL::VARCHAR
    FROM (
        SELECT impression_id
        FROM candidates
        GROUP BY impression_id
        HAVING COUNT(*) != MIN(candidate_count) OR MIN(candidate_count) != MAX(candidate_count)
    )
    UNION ALL
    SELECT 'invalid_label_rows',
           COUNT(*) FILTER (WHERE label IS NULL OR label NOT IN (0, 1))::DOUBLE, NULL::VARCHAR
    FROM candidates
    UNION ALL
    SELECT 'missing_article_metadata', COUNT(*) FILTER (WHERE published_time IS NULL)::DOUBLE,
           NULL::VARCHAR
    FROM candidates
    UNION ALL
    SELECT 'future_article_rows',
           COUNT(*) FILTER (WHERE published_time > impression_time)::DOUBLE, NULL::VARCHAR
    FROM candidates
    UNION ALL
    SELECT 'null_impression_id_rate',
           COUNT(*) FILTER (WHERE impression_id IS NULL)::DOUBLE / NULLIF(COUNT(*), 0), NULL::VARCHAR
    FROM candidates
    UNION ALL
    SELECT 'null_user_id_rate',
           COUNT(*) FILTER (WHERE user_id IS NULL)::DOUBLE / NULLIF(COUNT(*), 0), NULL::VARCHAR
    FROM candidates
    UNION ALL
    SELECT 'null_article_id_rate',
           COUNT(*) FILTER (WHERE article_id IS NULL)::DOUBLE / NULLIF(COUNT(*), 0), NULL::VARCHAR
    FROM candidates
    UNION ALL
    SELECT 'null_impression_time_rate',
           COUNT(*) FILTER (WHERE impression_time IS NULL)::DOUBLE / NULLIF(COUNT(*), 0), NULL::VARCHAR
    FROM candidates
    UNION ALL
    SELECT 'null_published_time_rate',
           COUNT(*) FILTER (WHERE published_time IS NULL)::DOUBLE / NULLIF(COUNT(*), 0), NULL::VARCHAR
    FROM candidates
    UNION ALL
    SELECT 'null_label_rate',
           COUNT(*) FILTER (WHERE label IS NULL)::DOUBLE / NULLIF(COUNT(*), 0), NULL::VARCHAR
    FROM candidates
    UNION ALL
    SELECT 'min_impression_time', NULL::DOUBLE, CAST(MIN(impression_time) AS VARCHAR)
    FROM candidates
    UNION ALL
    SELECT 'max_impression_time', NULL::DOUBLE, CAST(MAX(impression_time) AS VARCHAR)
    FROM candidates
)
SELECT metric, numeric_value, text_value
FROM quality
ORDER BY metric;
