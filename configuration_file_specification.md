# Configuration File Specification
----

## Domain specification:

- Domains are specified using the following type of line:

  ```
  d [num_attrs] [attr_1_domain_size] [attr_2_domain_size] ... [attr_n_domain_size]
  ```

  *This line must appear first, ignoring comments in any configuration file.*

## Agent specification:

- Each agent is enclosed in an agent block:

  ```
  AGENT
  ...
  END
  ```

- Each agent must specify how many examples are to be made inside their agent block:

  ```
  Size: [N]
  ```

- Each agent must have a type specification inside their agent block:

  ```
  Type: RPF
  ```

  *Valid agent types are listed below.*

- Each representation may specify additional configuration as needed.

| Type                       | Additional Fields           |
|----------------------------|-----------------------------|
| Weighted Average           | N/A                         |
| Penalty Logic              | Formulas, Clauses, Literals |
| LPM                        | N/A                         |
| Ranking Preference Formula | Clauses, Literals, Ranks    |

## Comments

- Comments are supported and specified by a '#' at the beginning of the line.

- Inline comments are not allowed.
