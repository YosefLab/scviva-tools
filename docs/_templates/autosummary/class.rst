{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block attributes %}
{% if attributes %}
.. rubric:: Attributes table

.. autosummary::
{% for item in attributes %}
    {%- if (fullname ~ '.' ~ item) not in torch_inherited %}
    ~{{ objname }}.{{ item }}
    {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if methods %}
.. rubric:: Methods table

.. autosummary::
{% for item in methods %}
    {%- if item != '__init__' and (fullname ~ '.' ~ item) not in torch_inherited %}
    ~{{ objname }}.{{ item }}
    {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes_documentation %}
{% if attributes %}
.. rubric:: Attributes

{% for item in attributes %}
{%- if (fullname ~ '.' ~ item) not in torch_inherited %}

.. autoattribute:: {{ [objname, item] | join(".") }}
{%- endif -%}
{%- endfor %}

{% endif %}
{% endblock %}

{% block methods_documentation %}
{% if methods %}
.. rubric:: Methods

{% for item in methods %}
{%- if item != '__init__' and (fullname ~ '.' ~ item) not in torch_inherited %}

.. automethod:: {{ [objname, item] | join(".") }}
{%- endif -%}
{%- endfor %}

{% endif %}
{% endblock %}
