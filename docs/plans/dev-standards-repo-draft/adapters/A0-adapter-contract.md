# A0 — Adapter Contract

## Rule

An adapter may translate the standards into a specific tool or ecosystem, but it may not redefine the core.

## Adapter responsibilities

- show how core rules are applied with a specific tool
- add operational details that only matter because that tool is in use
- stay replaceable if the tool changes later

## Adapter constraints

- must not contradict the core
- must not quietly introduce universal rules
- must not assume its tool exists in every project

## Adapter template

Each adapter should define:

- when to activate it
- what core rules it operationalizes
- what tool-specific behaviors it adds
- what it intentionally does not cover

