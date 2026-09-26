import streamlit as st
import pandas as pd
import pickle
import folium
import requests

def html_block(raw):
    """
    Strip leading whitespace from every line of an HTML string.

    Markdown treats any line indented 4+ spaces as a code block.
    Because this file's HTML is written with nested indentation
    (matching the Python code's own indentation), a plain
    textwrap.dedent() only removes the *common* leading whitespace
    and leaves inner tags still indented 4+ spaces - which still
    gets rendered as a code block. Stripping every line individually
    avoids that regardless of nesting depth.
    """
    lines = raw.strip("\n").splitlines()
    return "\n".join(line.strip() for line in lines)


from streamlit_folium import st_folium
from streamlit_searchbox import st_searchbox
from math import radians, sin, cos, sqrt, atan2


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Taxi Trip Price Prediction",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD PICKLE FILE
# =========================================================
# The existing trained Polynomial Regression pipeline is loaded
# as-is and is never retrained or modified by this file.

with open("trip_price_model.pkl", "rb") as file:
    model_data = pickle.load(file)

le = model_data["label_encoder"]
scaler = model_data["scaler"]
poly = model_data["poly_features"]
model = model_data["model"]
feature_columns = model_data["feature_columns"]


# =========================================================
# LOCATIONIQ API KEY
# =========================================================
# Read from .streamlit/secrets.toml - never hardcoded here.
# See the setup notes delivered alongside this file for how to
# get a free key and where exactly to put it.

LOCATIONIQ_API_KEY = st.secrets.get("LOCATIONIQ_API_KEY", "")

if not LOCATIONIQ_API_KEY:

    st.error(
        "LocationIQ API key not found. Add it to "
        "`.streamlit/secrets.toml` as "
        "`LOCATIONIQ_API_KEY = \"your_key_here\"` and restart the app. "
        "See the setup notes for how to get a free key."
    )

    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "pickup" not in st.session_state:
    st.session_state.pickup = None

if "drop" not in st.session_state:
    st.session_state.drop = None

if "pickup_name" not in st.session_state:
    st.session_state.pickup_name = "Bhopal, Madhya Pradesh, India"

if "drop_name" not in st.session_state:
    st.session_state.drop_name = "Rani Kamalapati Railway Station, Bhopal"

if "last_map_click" not in st.session_state:
    st.session_state.last_map_click = None

if "route_info" not in st.session_state:
    st.session_state.route_info = None


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    html_block(
    """
    <style>

    /* =========================
       MAIN PAGE
    ========================= */

    .stApp {
        background-color: #f4f7fb;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =========================
       HEADER
    ========================= */

    .header-box {
        background: linear-gradient(
            100deg,
            #10234a,
            #172554
        );

        padding: 30px 38px;

        border-radius: 18px;

        margin-bottom: 25px;

        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.15);
    }

    .header-title {
        color: #ffffff !important;
        font-size: 40px;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }

    .header-title .yellow {
        color: #fbbf24 !important;
    }

    .header-subtitle {
        color: #e2e8f0 !important;
        font-size: 17px;
        margin-top: 8px;
    }

    .header-description {
        color: #cbd5e1 !important;
        font-size: 14px;
        margin-top: 12px;
    }


    /* =========================
       CARD
    ========================= */

    .section-card {
        background: #ffffff;

        padding: 24px;

        border-radius: 18px;

        border: 1px solid #e5e7eb;

        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.07);

        margin-bottom: 20px;
    }

    .section-title {
        color: #172554 !important;
        font-size: 24px;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: #64748b !important;
        font-size: 14px;
        margin-bottom: 5px;
    }


    /* =========================
       LABELS
    ========================= */

    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label {
        color: #172554 !important;
        font-size: 14px !important;
        font-weight: 650 !important;
    }


    /* =========================
       TEXT BOX
    ========================= */

    .stTextInput input {
        background-color: #ffffff !important;
        color: #172554 !important;

        border: 1.5px solid #d5dce8 !important;

        border-radius: 9px !important;

        min-height: 48px !important;

        font-size: 15px !important;

        padding-left: 14px !important;
    }


    /* =========================
       NUMBER BOX
    ========================= */

    .stNumberInput input {
        background-color: #ffffff !important;
        color: #172554 !important;

        border: 1.5px solid #d5dce8 !important;

        border-radius: 9px !important;

        min-height: 48px !important;

        font-size: 15px !important;
    }


    /* =========================
       SELECT BOX
    ========================= */

    .stSelectbox [data-baseweb="select"] > div {
        background-color: #ffffff !important;

        color: #172554 !important;

        border: 1.5px solid #d5dce8 !important;

        border-radius: 9px !important;

        min-height: 48px !important;
    }

    .stSelectbox [data-baseweb="select"] span {
        color: #172554 !important;
    }


    /* =========================
       INPUT FOCUS
    ========================= */

    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: #2563eb !important;

        box-shadow:
            0 0 0 2px rgba(37, 99, 235, 0.12) !important;
    }


    /* =========================
       PREDICT BUTTON
    ========================= */

    .stButton > button {
        width: 100%;

        height: 52px;

        background: linear-gradient(
            90deg,
            #0ea5e9,
            #2563eb
        ) !important;

        color: #ffffff !important;

        border: none !important;

        border-radius: 9px !important;

        font-size: 17px !important;

        font-weight: 700 !important;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #0284c7,
            #1d4ed8
        ) !important;
    }


    /* =========================
       RESULT CARD
    ========================= */

    .result-card {
        background: linear-gradient(
            135deg,
            #eafff0,
            #f5fff8
        );

        border: 1px solid #c7efd4;

        padding: 25px;

        border-radius: 18px;

        text-align: center;

        margin-top: 25px;
    }

    .result-title {
        color: #172554 !important;
        font-size: 20px;
        font-weight: 700;
    }

    .result-price {
        color: #16a34a !important;
        font-size: 42px;
        font-weight: 800;
        margin-top: 5px;
    }

    .result-description {
        color: #64748b !important;
        font-size: 14px;
    }


    /* =========================
       SIDEBAR
    ========================= */

    section[data-testid="stSidebar"] {
        background-color: #101b33 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }


    /* =========================
       INFO BOX
    ========================= */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    </style>
    """
    ),
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        html_block(
        """
        <div style="
            font-size:22px;
            font-weight:800;
            padding:10px 0 25px 0;
            color:white;
        ">
            🚕 Taxi Fare Predictor
        </div>
        """
        ),
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🏠 Home")
    st.markdown("### 📊 Prediction")
    st.markdown("### ℹ️ About")
    st.markdown("### 🗄️ Dataset Info")

    st.markdown("---")

    st.info(
        "Search for a pickup and drop location above the map "
        "(or click the map directly) and fill in your trip "
        "details to get an estimated fare."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    html_block(
    """
    <div class="header-box">

        <div class="header-title">
            🚕 <span class="yellow">Taxi</span>
            Trip Price Prediction
        </div>

        <div class="header-subtitle">
            Predict your taxi fare easily using Machine Learning
        </div>

        <div class="header-description">
            Search for a pickup and drop location below
            and fill other details to get an estimated fare.
        </div>

    </div>
    """
    ),
    unsafe_allow_html=True
)


# =========================================================
# HAVERSINE DISTANCE  (FALLBACK ONLY)
# =========================================================
# Straight-line ("as the crow flies") distance between two
# points. This is NEVER treated as road distance in the UI - it
# is only used, and clearly labelled, when the LocationIQ
# Directions (routing) API below is unavailable.

def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)

    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        +
        cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c


# =========================================================
# LOCATIONIQ GEOCODING / ROUTING HELPERS
# =========================================================
#
# LocationIQ is an OpenStreetMap-based provider that is properly
# licensed for autocomplete/typeahead use (unlike the public
# Nominatim server, whose usage policy does not allow that).
# Free tier: 5,000 requests/day, 2 requests/second - plenty for
# a portfolio/demo project. See the setup notes for how to get
# a free API key.
#
# Every function below is wrapped in st.cache_data so identical
# queries (same search text, or the same pickup/drop pair) are
# served from cache instead of re-hitting the API - this is the
# "avoid unnecessary API calls" / caching requirement.

LOCATIONIQ_SEARCH_BASE = "https://api.locationiq.com/v1"
LOCATIONIQ_ROUTING_BASE = "https://us1.locationiq.com/v1"


@st.cache_data(ttl=600, show_spinner=False)
def locationiq_autocomplete(query, api_key, limit=5):
    """Return up to `limit` (label, lat, lon) matches for partial text."""

    if not query or len(query.strip()) < 3:
        return []

    try:
        response = requests.get(
            f"{LOCATIONIQ_SEARCH_BASE}/autocomplete",
            params={
                "key": api_key,
                "q": query,
                "limit": limit,
                "format": "json"
            },
            timeout=5
        )
        response.raise_for_status()

        return [
            (
                result.get("display_name", query),
                float(result["lat"]),
                float(result["lon"])
            )
            for result in response.json()
        ]

    except Exception:
        # Network error, invalid key, no matches, timeout, etc.
        # Fail quietly - the search box just shows no suggestions.
        return []


@st.cache_data(ttl=600, show_spinner=False)
def locationiq_reverse(lat, lon, api_key):
    """Turn map-click coordinates into a human-readable address."""

    try:
        response = requests.get(
            f"{LOCATIONIQ_SEARCH_BASE}/reverse",
            params={
                "key": api_key,
                "lat": lat,
                "lon": lon,
                "format": "json"
            },
            timeout=5
        )
        response.raise_for_status()

        return response.json().get(
            "display_name",
            f"Selected Location ({lat:.5f}, {lon:.5f})"
        )

    except Exception:
        return f"Selected Location ({lat:.5f}, {lon:.5f})"


@st.cache_data(ttl=600, show_spinner=False)
def locationiq_directions(pickup, drop, api_key):
    """
    Road route between two (lat, lon) points via LocationIQ's
    Directions API (an OSRM-based driving router).

    Returns a dict:
        {
            "distance_km":  road distance in kilometers,
            "duration_min": estimated driving time in minutes,
            "route_coords": [(lat, lon), ...] tracing the road path
        }
    or None if the routing request fails for any reason (network
    error, invalid response, no route found, timeout, etc). Callers
    must fall back to haversine_distance() when this returns None,
    and must label that fallback clearly in the UI - never silently
    present it as road distance.
    """

    try:
        # LocationIQ's routing endpoints take coordinates as
        # "longitude,latitude" - the opposite of the usual order.
        coords = (
            f"{pickup[1]},{pickup[0]};"
            f"{drop[1]},{drop[0]}"
        )

        response = requests.get(
            f"{LOCATIONIQ_ROUTING_BASE}/directions/driving/{coords}",
            params={
                "key": api_key,
                "overview": "full",
                "geometries": "geojson",
                "steps": "false"
            },
            timeout=8
        )
        response.raise_for_status()

        data = response.json()

        if data.get("code") != "ok" or not data.get("routes"):
            return None

        route = data["routes"][0]

        route_coords = [
            (lat, lon)
            for lon, lat in route["geometry"]["coordinates"]
        ]

        return {
            "distance_km": route["distance"] / 1000,
            "duration_min": route["duration"] / 60,
            "route_coords": route_coords
        }

    except Exception:
        return None


# =========================================================
# PICKUP / DROP LIVE SEARCH  (feeds st_searchbox)
# =========================================================
#
# st_searchbox calls this on every keystroke (after a short
# debounce) and shows the returned labels in a live dropdown,
# exactly like Google Maps' search box. Each option is returned
# as a (label, (label, lat, lon)) pair - the label is what's
# shown in the dropdown, and the (label, lat, lon) tuple is what
# we get back once the user clicks a suggestion.

def _pickup_search_fn(searchterm):

    return [
        (label, (label, lat, lon))
        for label, lat, lon in locationiq_autocomplete(
            searchterm,
            LOCATIONIQ_API_KEY
        )
    ]


def _drop_search_fn(searchterm):

    return [
        (label, (label, lat, lon))
        for label, lat, lon in locationiq_autocomplete(
            searchterm,
            LOCATIONIQ_API_KEY
        )
    ]


# =========================================================
# MAIN LAYOUT
# =========================================================

left, right = st.columns(
    [1, 1.25],
    gap="large"
)


# =========================================================
# LEFT SIDE - TRIP DETAILS
# =========================================================

with left:

    # ---------- CARD HEADING ----------

    st.markdown(
        html_block(
        """
        <div class="section-card">

            <div class="section-title">
                🧳 Trip Details
            </div>

            <div class="section-subtitle">
                Search for a place below, or select locations on the map
            </div>

        </div>
        """
        ),
        unsafe_allow_html=True
    )


    # ---------- PICKUP ----------
    # The widget's key is tied to the current pickup name, so that
    # when a map click changes it (see MAP CLICK section below) the
    # searchbox resets and displays the new address. The key stays
    # constant while the user is simply typing/searching.

    st.markdown(
        "#### 📍 Pickup Location"
    )

    pickup_selection = st_searchbox(
        _pickup_search_fn,
        key=f"pickup_searchbox::{st.session_state.pickup_name}",
        placeholder="Search for a place, address or landmark...",
        default_searchterm=st.session_state.pickup_name,
        debounce=400
    )

    if pickup_selection:

        label, lat, lon = pickup_selection

        if (lat, lon) != st.session_state.pickup:

            st.session_state.pickup = (lat, lon)
            st.session_state.pickup_name = label

    pickup_text = st.session_state.pickup_name


    # ---------- DROP ----------

    st.markdown(
        "#### 📍 Drop Location"
    )

    drop_selection = st_searchbox(
        _drop_search_fn,
        key=f"drop_searchbox::{st.session_state.drop_name}",
        placeholder="Search for a place, address or landmark...",
        default_searchterm=st.session_state.drop_name,
        debounce=400
    )

    if drop_selection:

        label, lat, lon = drop_selection

        if (lat, lon) != st.session_state.drop:

            st.session_state.drop = (lat, lon)
            st.session_state.drop_name = label

    drop_text = st.session_state.drop_name


    # ---------- ROUTE / DISTANCE / DURATION ----------
    # Distance and duration are auto-computed from the selected
    # pickup/drop pair via LocationIQ's road routing whenever
    # possible. Both fields stay editable (pre-filled with the
    # computed value) in case the user wants to explore a
    # different scenario. Their widget keys are tied to the
    # current pickup/drop pair, so the pre-filled value refreshes
    # whenever the selected locations change.

    st.markdown(
        "#### 🛣️ Route & Distance"
    )

    route_pair_key = f"{st.session_state.pickup}|{st.session_state.drop}"
    st.session_state.route_info = None

    if st.session_state.pickup is None or st.session_state.drop is None:

        st.info(
            "Search and select both a pickup and a drop location "
            "above to auto-calculate distance and duration."
        )

        default_distance = 5.0
        default_duration = 20.0

    elif st.session_state.pickup == st.session_state.drop:

        st.error(
            "Pickup and drop locations cannot be the same. "
            "Please choose a different drop location."
        )

        default_distance = 5.0
        default_duration = 20.0

    else:

        route_info = locationiq_directions(
            st.session_state.pickup,
            st.session_state.drop,
            LOCATIONIQ_API_KEY
        )

        if route_info:

            st.session_state.route_info = route_info

            default_distance = round(route_info["distance_km"], 2)
            default_duration = round(route_info["duration_min"], 0)

            st.success(
                f"🛣️ Road distance: {default_distance:.2f} km   •   "
                f"⏱️ Est. duration: {default_duration:.0f} min   "
                f"(road route via LocationIQ)"
            )

        else:

            default_distance = round(
                haversine_distance(
                    st.session_state.pickup[0],
                    st.session_state.pickup[1],
                    st.session_state.drop[0],
                    st.session_state.drop[1]
                ),
                2
            )
            default_duration = 20.0

            st.warning(
                "Route could not be calculated right now, so this is "
                f"a straight-line estimate, not the real road "
                f"distance: {default_distance:.2f} km. Please enter "
                "the trip duration manually below."
            )

    col1, col2 = st.columns(2)

    with col1:

        distance = st.number_input(
            "Trip Distance (km)",
            min_value=0.1,
            value=float(default_distance),
            step=0.1,
            format="%.2f",
            key=f"distance_input::{route_pair_key}"
        )

    with col2:

        trip_duration = st.number_input(
            "Trip Duration (Minutes)",
            min_value=1.0,
            value=float(default_duration),
            step=1.0,
            format="%.2f",
            key=f"duration_input::{route_pair_key}"
        )


    # ---------- PASSENGER + DAY ----------

    col1, col2 = st.columns(2)

    with col1:

        passenger_count = st.selectbox(
            "👥 Passenger Count",
            [1, 2, 3, 4]
        )

    with col2:

        day_options = list(
            le.classes_
        )

        day_of_week = st.selectbox(
            "📅 Day of Week",
            day_options
        )


    # ---------- TIME + TRAFFIC ----------

    col1, col2 = st.columns(2)

    with col1:

        time_of_day = st.selectbox(
            "🕐 Time of Day",
            [
                "Morning",
                "Afternoon",
                "Evening",
                "Night"
            ]
        )

    with col2:

        traffic = st.selectbox(
            "🚗 Traffic Conditions",
            [
                "Low",
                "Medium",
                "High"
            ]
        )


    # ---------- WEATHER ----------

    weather = st.selectbox(
        "☁️ Weather",
        [
            "Clear",
            "Rain",
            "Cloudy"
        ]
    )


    # ---------- BASE FARE + PER MINUTE ----------

    col1, col2 = st.columns(2)

    with col1:

        base_fare = st.number_input(
            "Base Fare",
            min_value=0.0,
            value=3.50,
            step=0.10,
            format="%.2f"
        )

    with col2:

        per_minute_rate = st.number_input(
            "Per Minute Rate",
            min_value=0.0,
            value=0.30,
            step=0.01,
            format="%.2f"
        )


    # ---------- PER KM RATE ----------

    per_km_rate = st.number_input(
        "Per Km Rate",
        min_value=0.0,
        value=1.20,
        step=0.01,
        format="%.2f"
    )


    st.markdown("<br>", unsafe_allow_html=True)


    # ---------- PREDICT BUTTON ----------

    predict_button = st.button(
        "🚕  Predict Fare",
        use_container_width=True
    )


# =========================================================
# RIGHT SIDE - MAP
# =========================================================

with right:

    st.markdown(
        html_block(
        """
        <div class="section-card">

            <div class="section-title">
                🗺️ Pickup &amp; Drop Location
            </div>

            <div class="section-subtitle">
                Selected via search above - or click the map directly
            </div>

        </div>
        """
        ),
        unsafe_allow_html=True
    )


    # ---------- CREATE MAP ----------
    # Centers on the selected pickup/drop point, like Google Maps
    # jumping to a searched or clicked location.

    if st.session_state.pickup and st.session_state.drop:

        map_center = [
            (st.session_state.pickup[0] + st.session_state.drop[0]) / 2,
            (st.session_state.pickup[1] + st.session_state.drop[1]) / 2
        ]
        map_zoom = 12

    elif st.session_state.pickup:

        map_center = list(st.session_state.pickup)
        map_zoom = 14

    elif st.session_state.drop:

        map_center = list(st.session_state.drop)
        map_zoom = 14

    else:

        map_center = [23.2599, 77.4126]
        map_zoom = 12

    m = folium.Map(
        location=map_center,
        zoom_start=map_zoom,
        tiles="OpenStreetMap"
    )


    # ---------- PICKUP MARKER ----------

    if st.session_state.pickup:

        folium.Marker(
            location=st.session_state.pickup,

            popup="Pickup Location",

            tooltip="Pickup",

            icon=folium.Icon(
                color="green",
                icon="map-marker"
            )

        ).add_to(m)


    # ---------- DROP MARKER ----------

    if st.session_state.drop:

        folium.Marker(
            location=st.session_state.drop,

            popup="Drop Location",

            tooltip="Drop",

            icon=folium.Icon(
                color="red",
                icon="map-marker"
            )

        ).add_to(m)


    # ---------- ROUTE LINE ----------
    # Uses the real road-route geometry from LocationIQ when
    # available; otherwise draws a dashed straight line, clearly
    # distinct in style, since it is only an approximation.

    if st.session_state.pickup and st.session_state.drop:

        if (
            st.session_state.route_info
            and st.session_state.route_info.get("route_coords")
        ):

            folium.PolyLine(
                st.session_state.route_info["route_coords"],
                color="#2563eb",
                weight=5,
                opacity=0.85
            ).add_to(m)

        else:

            folium.PolyLine(
                [
                    st.session_state.pickup,
                    st.session_state.drop
                ],
                color="#94a3b8",
                weight=4,
                opacity=0.75,
                dash_array="8, 8"
            ).add_to(m)

        # Frame both markers (and the route) nicely in view.
        m.fit_bounds(
            [
                st.session_state.pickup,
                st.session_state.drop
            ]
        )


    # ---------- DISPLAY MAP ----------

    map_data = st_folium(
        m,
        width=None,
        height=520,
        returned_objects=[
            "last_clicked"
        ]
    )


    # =====================================================
    # MAP CLICK  (optional - search above is the primary way
    # to select locations, but clicking the map still works)
    # =====================================================

    if map_data["last_clicked"]:

        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]

        current_click = (
            round(lat, 6),
            round(lon, 6)
        )


        # Prevent same click from being processed repeatedly

        if current_click != st.session_state.last_map_click:

            st.session_state.last_map_click = current_click

            place_name = locationiq_reverse(lat, lon, LOCATIONIQ_API_KEY)


            # First click = Pickup

            if st.session_state.pickup is None:

                st.session_state.pickup = (
                    lat,
                    lon
                )

                st.session_state.pickup_name = place_name


            # Second click = Drop

            elif st.session_state.drop is None:

                st.session_state.drop = (
                    lat,
                    lon
                )

                st.session_state.drop_name = place_name


            # Third click = New pickup

            else:

                st.session_state.pickup = (
                    lat,
                    lon
                )

                st.session_state.drop = None

                st.session_state.pickup_name = place_name

                st.session_state.drop_name = (
                    "Select drop location on map"
                )


            # Changing pickup_name/drop_name changes the searchbox
            # widgets' keys (see LEFT SIDE section), so on rerun they
            # rebuild fresh and display this new address.
            st.rerun()


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    if st.session_state.pickup is None:

        st.error("Please select a pickup location.")

    elif st.session_state.drop is None:

        st.error("Please select a drop location.")

    elif st.session_state.pickup == st.session_state.drop:

        st.error(
            "Pickup and drop locations cannot be the same. "
            "Please choose a different drop location."
        )

    else:

        try:

            # =================================================
            # CREATE INPUT DATA
            # =================================================

            input_data = pd.DataFrame({

                "Trip_Distance_km": [
                    distance
                ],

                "Time_of_Day": [
                    time_of_day
                ],

                "Day_of_Week": [
                    day_of_week
                ],

                "Passenger_Count": [
                    passenger_count
                ],

                "Traffic_Conditions": [
                    traffic
                ],

                "Weather": [
                    weather
                ],

                "Base_Fare": [
                    base_fare
                ],

                "Per_Km_Rate": [
                    per_km_rate
                ],

                "Per_Minute_Rate": [
                    per_minute_rate
                ],

                "Trip_Duration_Minutes": [
                    trip_duration
                ]
            })


            # =================================================
            # LABEL ENCODING
            # =================================================

            input_data["Day_of_Week"] = le.transform(
                input_data["Day_of_Week"]
            )


            # =================================================
            # ONE-HOT ENCODING
            # =================================================

            input_data = pd.get_dummies(
                input_data,

                columns=[
                    "Time_of_Day",
                    "Traffic_Conditions",
                    "Weather"
                ],

                dtype=int
            )


            # =================================================
            # SAME COLUMNS AS TRAINING
            # =================================================

            input_data = input_data.reindex(
                columns=feature_columns,
                fill_value=0
            )


            # =================================================
            # SCALING
            # =================================================

            input_scaled = scaler.transform(
                input_data
            )


            # =================================================
            # POLYNOMIAL FEATURES
            # =================================================

            input_poly = poly.transform(
                input_scaled
            )


            # =================================================
            # PREDICTION
            # =================================================

            prediction = model.predict(
                input_poly
            )[0]


            prediction = max(
                0,
                prediction
            )


            # =================================================
            # RESULT
            # =================================================

            st.markdown(
                html_block(
                f"""
                <div class="result-card">

                    <div class="result-title">
                        💰 Estimated Trip Price
                    </div>

                    <div class="result-price">
                        ₹ {prediction:.2f}
                    </div>

                    <div class="result-description">
                        Based on the provided trip details
                    </div>

                </div>
                """
                ),
                unsafe_allow_html=True
            )


            # =================================================
            # TRIP SUMMARY
            # =================================================

            st.markdown(
                html_block(
                """
                <div class="section-card"
                     style="margin-top:20px;">

                    <div class="section-title">
                        📋 Trip Summary
                    </div>

                </div>
                """
                ),
                unsafe_allow_html=True
            )


            col1, col2 = st.columns(2)


            with col1:

                st.write(
                    f"📍 **Pickup:** {pickup_text}"
                )

                st.write(
                    f"📍 **Drop:** {drop_text}"
                )

                distance_source = (
                    "road route"
                    if st.session_state.route_info
                    else "straight-line estimate"
                )

                st.write(
                    f"🛣️ **Distance:** {distance:.2f} km "
                    f"({distance_source})"
                )

                st.write(
                    f"⏱️ **Trip Duration:** "
                    f"{trip_duration:.0f} minutes"
                )


            with col2:

                st.write(
                    f"👥 **Passenger Count:** "
                    f"{passenger_count}"
                )

                st.write(
                    f"📅 **Day:** {day_of_week}"
                )

                st.write(
                    f"🕐 **Time:** {time_of_day}"
                )

                st.write(
                    f"🚗 **Traffic:** {traffic}"
                )

                st.write(
                    f"☁️ **Weather:** {weather}"
                )


        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

            st.info(
                "Please check that trip_price_model.pkl "
                "was created from the same notebook/model."
            )
