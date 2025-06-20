Hey! I made an improvement to the excercise/challenge. I decided to create a yaml file that will add specific context about the tables to make the generation even smarter some features:
dialect: wanted to provide which dialect or flavor of SQL since that can change.
specific tables with a table description and column descriptions as well.
mark if a column is sensitive. Sometimes you have PII information you dont want to pass to the LLM or want it to use.
mark if a column is a dimension(something to group by) or an metric (something to perform calculations on)
mark if a column has aggregations (specfy the kinds of aggregations it can do)
mark if a column has additional meanings (add additional context to what a column can mean)
examples questions and queries to return
relationships (define the how the tables can be joined)
rules to follow.
This approach makes it way more specific and flexible and powerful.
I had to create a new custom sql query chain to leverage this, so not using the create_sql_query_chain anymore. This gave me much more control.
A good test was “Show me total revenue by product over the last year” (Note that if you run the query it will return 0 results because data is outside of the 12 month window). But I added a rule to give it awareness that the max date was in 2020 so dont use now.
