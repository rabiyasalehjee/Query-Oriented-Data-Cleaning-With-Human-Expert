<?php
$servername = "localhost"; // Server name, usually "localhost" for local development
$username = "root"; // MySQL username
$password = ""; // MySQL password, empty by default for local development
$database = "query_data_cleaning"; // Your MySQL database name

// Create a connection to the MySQL server
$conn = new mysqli($servername, $username, $password, $database);

// Check the connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

?>
