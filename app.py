import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium

from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Suraksha - Mahila Suraksha Route",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# DATASET
# ============================================================

DATA_FILE = "womens_safety_filtered_data.csv"


# ============================================================
# SESSION STATE
# ============================================================

if "route_results" not in st.session_state:
    st.session_state.route_results = None

if "source_location" not in st.session_state:
    st.session_state.source_location = None

if "destination_location" not in st.session_state:
    st.session_state.destination_location = None

if "source_name" not in st.session_state:
    st.session_state.source_name = ""

if "destination_name" not in st.session_state:
    st.session_state.destination_name = ""

if "selected_time" not in st.session_state:
    st.session_state.selected_time = ""


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <h1 style="
        text-align:center;
        font-size:42px;
        margin-bottom:5px;
    ">
        🛡️ Suraksha - Mahila Suraksha Route
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="
        text-align:center;
        font-size:20px;
        margin-top:0px;
    ">
        Women Safety Risk Prediction System
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="
        text-align:center;
        font-size:16px;
    ">
        Find a relatively safer route using historical FIR crime data.
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET AUTOMATICALLY
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_FILE)

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return data


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "Dataset not found."
    )

    st.info(
        f"""
Keep the CSV file in the same folder as app.py.

Required filename:

{DATA_FILE}
"""
    )

    st.stop()

except Exception as error:

    st.error(
        f"Error loading dataset: {error}"
    )

    st.stop()


# ============================================================
# FIND REQUIRED COLUMNS
# ============================================================

def find_column(possible_names):

    for column in possible_names:

        if column in df.columns:
            return column

    return None


LAT_COL = find_column(
    [
        "latitude",
        "lat"
    ]
)

LON_COL = find_column(
    [
        "longitude",
        "lon",
        "lng"
    ]
)


# ============================================================
# CHECK LOCATION DATA
# ============================================================

if LAT_COL is None or LON_COL is None:

    st.error(
        "Latitude and longitude columns were not found in your CSV."
    )

    st.write(
        "Columns found in your dataset:"
    )

    st.write(
        list(df.columns)
    )

    st.stop()


# ============================================================
# CLEAN LOCATION DATA
# ============================================================

df[LAT_COL] = pd.to_numeric(
    df[LAT_COL],
    errors="coerce"
)

df[LON_COL] = pd.to_numeric(
    df[LON_COL],
    errors="coerce"
)

df = df.dropna(
    subset=[
        LAT_COL,
        LON_COL
    ]
)


# Keep coordinates that are reasonably within India

df = df[
    (df[LAT_COL] >= 6)
    &
    (df[LAT_COL] <= 38)
    &
    (df[LON_COL] >= 68)
    &
    (df[LON_COL] <= 98)
]


# ============================================================
# DATASET INFORMATION
# ============================================================

if len(df) == 0:

    st.error(
        "No valid latitude and longitude records were found."
    )

    st.stop()


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine(
    lat1,
    lon1,
    lat2,
    lon2
):

    radius = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = np.radians(
        lat2 - lat1
    )

    dlon = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2) ** 2
    )

    c = (
        2
        *
        np.arctan2(
            np.sqrt(a),
            np.sqrt(1 - a)
        )
    )

    return radius * c


# ============================================================
# GEOCODING
# ============================================================

def find_location(place):

    url = (
        "https://nominatim.openstreetmap.org/search"
    )

    params = {

        "q": f"{place}, Karnataka, India",

        "format": "json",

        "limit": 1
    }

    headers = {

        "User-Agent":
        "Suraksha-Mahila-Suraksha-Route"
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if not data:

            return None

        latitude = float(
            data[0]["lat"]
        )

        longitude = float(
            data[0]["lon"]
        )

        return (
            latitude,
            longitude
        )

    except Exception:

        return None


# ============================================================
# GET ROAD ROUTES
# ============================================================

def get_routes(
    source_lat,
    source_lon,
    destination_lat,
    destination_lon
):

    url = (
        "https://router.project-osrm.org/"
        "route/v1/driving/"
        f"{source_lon},{source_lat};"
        f"{destination_lon},{destination_lat}"
    )

    params = {

        "alternatives": "true",

        "overview": "full",

        "geometries": "geojson"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        if result.get("code") != "Ok":

            return []

        return result.get(
            "routes",
            []
        )

    except Exception:

        return []


# ============================================================
# CALCULATE HISTORICAL ROUTE RISK
# ============================================================

def calculate_route_risk(
    coordinates,
    travel_hour
):

    if not coordinates:

        return 0.0


    # Convert route points
    # from [longitude, latitude]
    # to [latitude, longitude]

    route_points = np.array(
        [
            [
                point[1],
                point[0]
            ]
            for point in coordinates
        ]
    )


    # Limit number of points
    # to improve speed

    if len(route_points) > 100:

        indexes = np.linspace(
            0,
            len(route_points) - 1,
            100
        ).astype(int)

        route_points = (
            route_points[indexes]
        )


    risk_values = []


    # Dataset coordinates

    crime_latitudes = (
        df[LAT_COL].values
    )

    crime_longitudes = (
        df[LON_COL].values
    )


    for latitude, longitude in route_points:

        distances = haversine(
            latitude,
            longitude,
            crime_latitudes,
            crime_longitudes
        )


        # Historical FIRs close to route

        crimes_1km = np.sum(
            distances <= 1
        )

        crimes_3km = np.sum(
            distances <= 3
        )


        # Historical crime density score

        point_score = (
            crimes_1km * 10
            +
            crimes_3km * 2
        )


        # Night travel factor

        if (
            travel_hour >= 20
            or
            travel_hour <= 5
        ):

            point_score *= 1.30


        risk_values.append(
            point_score
        )


    if not risk_values:

        return 0.0


    average_score = np.mean(
        risk_values
    )


    return float(
        average_score
    )


# ============================================================
# ASSIGN RELATIVE RISK LEVEL
# ============================================================

def assign_risk_levels(results):

    if len(results) == 1:

        results[0]["risk_level"] = (
            "LOW RISK"
        )

        return results


    # Sort from safest to riskiest

    results.sort(
        key=lambda x: x["risk_score"]
    )


    number_of_routes = len(results)


    # Two routes

    if number_of_routes == 2:

        results[0]["risk_level"] = (
            "LOW RISK"
        )

        results[1]["risk_level"] = (
            "HIGH RISK"
        )


    # Three or more routes

    else:

        results[0]["risk_level"] = (
            "LOW RISK"
        )

        results[-1]["risk_level"] = (
            "HIGH RISK"
        )


        for i in range(
            1,
            number_of_routes - 1
        ):

            results[i]["risk_level"] = (
                "MEDIUM RISK"
            )


    return results


# ============================================================
# USER INPUT
# ============================================================

st.markdown("---")

left_column, right_column = st.columns(2)


with left_column:

    source = st.text_input(
        "📍 Source",
        placeholder="Example: Davanagere Railway Station"
    )


with right_column:

    destination = st.text_input(
        "📍 Destination",
        placeholder="Example: Bapuji Nagar, Davanagere"
    )


travel_time = st.selectbox(
    "🕐 Travel Time",
    [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]
)


# Representative travel hours

travel_hours = {

    "Morning": 8,

    "Afternoon": 13,

    "Evening": 18,

    "Night": 22
}


travel_hour = travel_hours[
    travel_time
]


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔍 Predict Safer Route",
    use_container_width=True
):

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not source.strip():

        st.warning(
            "Please enter the source location."
        )

        st.stop()


    if not destination.strip():

        st.warning(
            "Please enter the destination."
        )

        st.stop()


    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    with st.spinner(
        "Finding source location..."
    ):

        source_location = find_location(
            source
        )


    if source_location is None:

        st.error(
            "Source location could not be found."
        )

        st.info(
            "Try entering a more specific location."
        )

        st.stop()


    # --------------------------------------------------------
    # DESTINATION
    # --------------------------------------------------------

    with st.spinner(
        "Finding destination location..."
    ):

        destination_location = find_location(
            destination
        )


    if destination_location is None:

        st.error(
            "Destination location could not be found."
        )

        st.info(
            "Try entering a more specific location."
        )

        st.stop()


    source_lat, source_lon = (
        source_location
    )

    destination_lat, destination_lon = (
        destination_location
    )


    # --------------------------------------------------------
    # FIND ROUTES
    # --------------------------------------------------------

    with st.spinner(
        "Finding road routes..."
    ):

        routes = get_routes(
            source_lat,
            source_lon,
            destination_lat,
            destination_lon
        )


    if not routes:

        st.error(
            "No road route was found."
        )

        st.stop()


    # --------------------------------------------------------
    # CALCULATE ROUTE SCORES
    # --------------------------------------------------------

    route_results = []


    for index, route in enumerate(
        routes
    ):

        coordinates = (
            route[
                "geometry"
            ][
                "coordinates"
            ]
        )


        risk_score = calculate_route_risk(
            coordinates,
            travel_hour
        )


        route_information = {

            "route_number":
                index + 1,

            "risk_score":
                risk_score,

            "risk_level":
                "",

            "distance":
                route["distance"] / 1000,

            "duration":
                route["duration"] / 60,

            "coordinates":
                coordinates
        }


        route_results.append(
            route_information
        )


    # --------------------------------------------------------
    # ASSIGN RISK LEVELS
    # --------------------------------------------------------

    route_results = assign_risk_levels(
        route_results
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    st.session_state.route_results = (
        route_results
    )

    st.session_state.source_location = (
        source_location
    )

    st.session_state.destination_location = (
        destination_location
    )

    st.session_state.source_name = (
        source
    )

    st.session_state.destination_name = (
        destination
    )

    st.session_state.selected_time = (
        travel_time
    )


# ============================================================
# DISPLAY STORED RESULTS
# ============================================================

if st.session_state.route_results is not None:

    results = (
        st.session_state.route_results
    )


    source_location = (
        st.session_state.source_location
    )


    destination_location = (
        st.session_state.destination_location
    )


    source = (
        st.session_state.source_name
    )


    destination = (
        st.session_state.destination_name
    )


    travel_time = (
        st.session_state.selected_time
    )


    source_lat, source_lon = (
        source_location
    )


    destination_lat, destination_lon = (
        destination_location
    )


    # ========================================================
    # SAFEST ROUTE
    # ========================================================

    safest_route = min(
        results,
        key=lambda x: x["risk_score"]
    )


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🛡️ Safety Prediction"
    )


    st.success(
        f"""
Recommended Route: Route
{safest_route['route_number']}

Risk Level: {safest_route['risk_level']}

Historical Risk Score:
{safest_route['risk_score']:.1f}/100
"""
    )


    # ========================================================
    # ROUTE COMPARISON
    # ========================================================

    st.subheader(
        "🛣️ Available Routes"
    )


    route_columns = st.columns(
        len(results)
    )


    for index, route in enumerate(
        results
    ):

        with route_columns[index]:

            st.markdown(
                f"### Route {route['route_number']}"
            )


            st.metric(
                "Risk Score",
                f"{route['risk_score']:.1f}/100"
            )


            if route["risk_level"] == "LOW RISK":

                st.success(
                    "🟢 LOW RISK"
                )


            elif route["risk_level"] == "MEDIUM RISK":

                st.warning(
                    "🟡 MEDIUM RISK"
                )


            else:

                st.error(
                    "🔴 HIGH RISK"
                )


            st.write(
                f"Distance: "
                f"{route['distance']:.2f} km"
            )


            st.write(
                f"Travel Time: "
                f"{route['duration']:.0f} minutes"
            )


    # ========================================================
    # MAP
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🗺️ Route Safety Map"
    )


    center_lat = (
        source_lat +
        destination_lat
    ) / 2


    center_lon = (
        source_lon +
        destination_lon
    ) / 2


    safety_map = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=13,
        control_scale=True
    )


    # ========================================================
    # SOURCE MARKER
    # ========================================================

    folium.Marker(

        location=[
            source_lat,
            source_lon
        ],

        popup=(
            f"<b>Source</b><br>"
            f"{source}"
        ),

        tooltip="Starting Point",

        icon=folium.Icon(
            color="blue",
            icon="play"
        )

    ).add_to(
        safety_map
    )


    # ========================================================
    # DESTINATION MARKER
    # ========================================================

    folium.Marker(

        location=[
            destination_lat,
            destination_lon
        ],

        popup=(
            f"<b>Destination</b><br>"
            f"{destination}"
        ),

        tooltip="Destination",

        icon=folium.Icon(
            color="green",
            icon="flag"
        )

    ).add_to(
        safety_map
    )


    # ========================================================
    # DRAW ROUTES
    # ========================================================

    for route in results:

        route_points = [

            [
                point[1],
                point[0]
            ]

            for point in route[
                "coordinates"
            ]
        ]


        # Determine route color

        if route["risk_level"] == "LOW RISK":

            route_color = "green"

        elif route["risk_level"] == "MEDIUM RISK":

            route_color = "orange"

        else:

            route_color = "red"


        # Safest route is thicker

        if (
            route["route_number"]
            ==
            safest_route["route_number"]
        ):

            line_weight = 9

        else:

            line_weight = 5


        folium.PolyLine(

            locations=route_points,

            color=route_color,

            weight=line_weight,

            opacity=0.85,

            tooltip=(
                f"Route {route['route_number']} | "
                f"{route['risk_level']} | "
                f"Risk Score: "
                f"{route['risk_score']:.1f}"
            ),

            popup=(
                f"<b>Route "
                f"{route['route_number']}</b><br>"
                f"Risk: "
                f"{route['risk_level']}<br>"
                f"Risk Score: "
                f"{route['risk_score']:.1f}/100<br>"
                f"Distance: "
                f"{route['distance']:.2f} km<br>"
                f"Travel Time: "
                f"{route['duration']:.0f} minutes"
            )

        ).add_to(
            safety_map
        )


    # ========================================================
    # MAP LEGEND
    # ========================================================

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 190px;
        z-index:9999;
        background-color:white;
        border:2px solid grey;
        border-radius:8px;
        padding:10px;
        font-size:14px;
    ">

    <b>Safety Level</b><br><br>

    <span style="color:green;">━━━</span>
    🟢 Low Risk<br>

    <span style="color:orange;">━━━</span>
    🟡 Medium Risk<br>

    <span style="color:red;">━━━</span>
    🔴 High Risk<br>

    <br>
    <b>Thicker route</b> =
    Recommended route

    </div>
    """


    safety_map.get_root().html.add_child(
        folium.Element(
            legend_html
        )
    )


    # ========================================================
    # DISPLAY MAP
    # ========================================================

    st_folium(
        safety_map,
        width=None,
        height=600
    )


    # ========================================================
    # FINAL RECOMMENDATION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🛡️ Suraksha Recommendation"
    )


    st.info(
        f"""
For your selected travel time of **{travel_time}**:

Recommended Route:
**Route {safest_route['route_number']}**

Risk Level:
**{safest_route['risk_level']}**

Historical Risk Score:
**{safest_route['risk_score']:.1f}/100**

Distance:
**{safest_route['distance']:.2f} km**

Estimated Travel Time:
**{safest_route['duration']:.0f} minutes**
"""
    )


    st.caption(
        "This system uses historical FIR patterns to "
        "calculate relative route risk. A low historical "
        "risk score does not guarantee future safety."
    )
    # ============================================================
# 6. 🚨 EMERGENCY SAFETY
# ============================================================

st.markdown("---")
st.subheader("🚨 Emergency Safety")

st.markdown("### 🆘 Emergency Support")

# ============================================================
# 🚨 SOS - ACTUAL CURRENT LOCATION
# ============================================================

sos_html = """
<button onclick="activateSOS()" style="
    width:100%;
    padding:20px;
    background-color:#d32f2f;
    color:white;
    border:none;
    border-radius:10px;
    font-size:24px;
    font-weight:bold;
    cursor:pointer;">
🚨 SOS - GET CURRENT LOCATION
</button>

<div id="sos_status"
     style="margin-top:15px;font-size:18px;font-weight:bold;
     line-height:1.7;">
</div>

<script>
function activateSOS() {

    const status = document.getElementById("sos_status");

    status.innerHTML = "📍 Getting your current location...";

    if (!navigator.geolocation) {
        status.innerHTML =
            "❌ GPS location is not supported by this browser.";
        return;
    }

    navigator.geolocation.getCurrentPosition(

        async function(position) {

            const latitude = position.coords.latitude;
            const longitude = position.coords.longitude;

            const mapLink =
                "https://www.google.com/maps?q="
                + latitude + "," + longitude;

            let placeName = "Current location";

            try {

                const response = await fetch(
                    "https://nominatim.openstreetmap.org/reverse"
                    + "?format=json"
                    + "&lat=" + latitude
                    + "&lon=" + longitude
                    + "&zoom=18"
                    + "&addressdetails=1"
                );

                const data = await response.json();

                if (data.address) {

                    placeName =
                        data.address.city ||
                        data.address.town ||
                        data.address.village ||
                        data.address.municipality ||
                        data.address.suburb ||
                        data.address.county ||
                        "Current location";
                }

            } catch (error) {

                placeName = "Current location";
            }

            status.innerHTML =
                "🚨 <b>SOS ACTIVATED!</b><br><br>" +
                "📍 <b>Current Place:</b> " +
                placeName + "<br><br>" +
                "Latitude: " +
                latitude.toFixed(6) + "<br>" +
                "Longitude: " +
                longitude.toFixed(6) + "<br><br>" +

                "<a href='" + mapLink +
                "' target='_blank'>" +
                "🗺️ Open Current Location in Google Maps" +
                "</a><br><br>" +

                "<a href='tel:112' style='font-size:20px;"
                + "font-weight:bold;'>📞 CALL 112</a>";
        },

        function(error) {

            status.innerHTML =
                "❌ Unable to get current location.<br>"
                + "Please allow location permission.";
        },

        {
            enableHighAccuracy: true,
            timeout: 15000,
            maximumAge: 0
        }
    );
}
</script>
"""

st.components.v1.html(sos_html, height=330)


# ============================================================
# 📞 EMERGENCY CONTACTS
# ============================================================

st.markdown("### 📞 Emergency Contacts")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <a href="tel:112">
        <button style="
        width:100%;padding:14px;background:#d32f2f;
        color:white;border:none;border-radius:8px;
        font-size:16px;font-weight:bold;">
        🚨 112 Emergency
        </button>
        </a>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <a href="tel:181">
        <button style="
        width:100%;padding:14px;background:#8e44ad;
        color:white;border:none;border-radius:8px;
        font-size:16px;font-weight:bold;">
        👩 181 Women Helpline
        </button>
        </a>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <a href="tel:1091">
        <button style="
        width:100%;padding:14px;background:#2980b9;
        color:white;border:none;border-radius:8px;
        font-size:16px;font-weight:bold;">
        👮 1091 Women Police
        </button>
        </a>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 📍 SHARE CURRENT LOCATION
# ============================================================

st.markdown("### 📍 Share Current Location")

share_html = """
<button onclick="shareLocation()" style="
    width:100%;
    padding:15px;
    border:none;
    border-radius:8px;
    font-size:18px;
    font-weight:bold;
    cursor:pointer;">
📍 Share My Current Location
</button>

<div id="share_status"
     style="margin-top:10px;font-size:16px;">
</div>

<script>
function shareLocation() {

    const status =
        document.getElementById("share_status");

    if (!navigator.geolocation) {
        status.innerHTML =
            "❌ GPS is not supported.";
        return;
    }

    status.innerHTML =
        "📍 Getting your current location...";

    navigator.geolocation.getCurrentPosition(

        function(position) {

            const lat = position.coords.latitude;
            const lon = position.coords.longitude;

            const url =
                "https://www.google.com/maps?q="
                + lat + "," + lon;

            if (navigator.share) {

                navigator.share({
                    title: "My Current Location",
                    text: "My current location:",
                    url: url
                });

            } else {

                navigator.clipboard.writeText(url);

                status.innerHTML =
                    "✅ Location link copied!<br>" +
                    "<a href='" + url +
                    "' target='_blank'>" +
                    "🗺️ Open Location" +
                    "</a>";
            }
        },

        function(error) {

            status.innerHTML =
                "❌ Location permission denied.";
        },

        {
            enableHighAccuracy: true,
            timeout: 15000,
            maximumAge: 0
        }
    );
}
</script>
"""

st.components.v1.html(share_html, height=120)


# ============================================================
# 🗺️ ROUTE-BASED POLICE & HOSPITAL SEARCH
# ============================================================

st.markdown("### 🗺️ Emergency Locations Along Your Route")

# Use the existing source and destination values from your app.
# These should be the same values the user selected for the route.

try:

    route_source = source
    route_destination = destination

except NameError:

    route_source = ""
    route_destination = ""


if route_source and route_destination:

    st.info(
        f"Emergency locations will be searched along the route: "
        f"**{route_source} → {route_destination}**"
    )

    # --------------------------------------------------------
    # 👮 POLICE STATIONS ALONG ROUTE
    # --------------------------------------------------------

    st.markdown("#### 👮 Police Stations Along Route")

    police_url = (
        "https://www.google.com/maps/search/"
        + requests.utils.quote(
            "police stations along route "
            + str(route_source)
            + " to "
            + str(route_destination)
        )
    )

    st.markdown(
        f"""
        <a href="{police_url}" target="_blank">
        <button style="
            width:100%;
            padding:15px;
            border:none;
            border-radius:8px;
            font-size:18px;
            font-weight:bold;
            cursor:pointer;">
        👮 View Police Stations Along Route
        </button>
        </a>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # 🏥 HOSPITALS ALONG ROUTE
    # --------------------------------------------------------

    st.markdown("#### 🏥 Hospitals Along Route")

    hospital_url = (
        "https://www.google.com/maps/search/"
        + requests.utils.quote(
            "hospitals along route "
            + str(route_source)
            + " to "
            + str(route_destination)
        )
    )

    st.markdown(
        f"""
        <a href="{hospital_url}" target="_blank">
        <button style="
            width:100%;
            padding:15px;
            border:none;
            border-radius:8px;
            font-size:18px;
            font-weight:bold;
            cursor:pointer;">
        🏥 View Hospitals Along Route
        </button>
        </a>
        """,
        unsafe_allow_html=True
    )

else:

    st.warning(
        "Please select a source and destination first "
        "to view emergency locations along the route."
    )


# ============================================================
# 👩 WOMEN HELPLINE
# ============================================================

st.markdown("### 👩 Women Helpline")

st.markdown(
    """
    <a href="tel:181">
    <button style="
        width:100%;
        padding:17px;
        background:#8e44ad;
        color:white;
        border:none;
        border-radius:8px;
        font-size:20px;
        font-weight:bold;
        cursor:pointer;">
    👩 CALL WOMEN HELPLINE — 181
    </button>
    </a>
    """,
    unsafe_allow_html=True
)
