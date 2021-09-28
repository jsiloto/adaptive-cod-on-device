val url = "http://my-json-feed"

val jsonObjectRequest = JsonObjectRequest(Request.Method.GET, url, null,
        Response.Listener { response ->
            textView.text = "Response: %s".format(response.toString())
        },
        Response.ErrorListener { error ->
            // TODO: Handle error
        }
)

// Access the RequestQueue through your singleton class.
MySingleton.getInstance(this).addToRequestQueue(jsonObjectRequest)