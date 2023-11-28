<?php
$connect = mysqli_connect("localhost", "root", "", "query_data_cleaning");
$query = "SELECT * FROM db_messy";
$result1 = mysqli_query($connect, $query);

$query_results = '';
$error_message = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $sql_query = $_POST['sql_query'];

    if (!empty($sql_query)) {
        $show_modal = false; // Initialize the flag to false

        // Check if the query is an attempt to add 'year' as a primary key
        if (stripos($sql_query, 'ALTER TABLE db_messy ADD PRIMARY KEY (year)') !== false) {
            // Check for duplicate or empty 'year' values
            $check_query = "SELECT COUNT(*) as count FROM db_messy WHERE year IS NULL OR year = ''";
            $check_result = mysqli_query($connect, $check_query);
            $row = mysqli_fetch_assoc($check_result);

            if ($row['count'] > 0) {
                $show_modal = true; // Set the flag to true
            } else {
                // Execute the ALTER TABLE query
                $result = mysqli_query($connect, $sql_query);
                if ($result) {
                    $query_results = 'Primary key for "year" column added successfully.';
                } else {
                    $error_message = 'Query execution failed: ' . mysqli_error($connect);
                }
            }
        } else {
            // Execute other queries
            $result = mysqli_query($connect, $sql_query);
            if ($result) {
                $query_results = '<table class="common-table table table-hover">';
                while ($row = mysqli_fetch_assoc($result)) {
                    $query_results .= '<tr>';
                    foreach ($row as $value) {
                        $query_results .= '<td' . (empty($value) ? ' class="empty-cell"' : '') . '>' . $value . '</td>';
                    }
                    $query_results .= '</tr>';
                }
                $query_results .= '</table>';
            } else {
                $error_message = 'Query execution failed: ' . mysqli_error($connect);
            }
        }
        
        // Add this condition to control the error message display
        if ($show_modal) {
            $error_message = 'The "year" column can\'t be created as a primary key because it contains empty or duplicate values. Do you want to modify the data?';
            
            // Code to display buttons
            /* echo '<div id="buttonsContainer">';
            echo '<button id="modifyDataButton">Yes, Modify</button>';
            echo '<button id="noModifyDataButton">No, Modify</button>';
            echo '</div>'; */
        }
    } else {
        $error_message = 'Query is empty. Please enter a valid SQL query.';
    }
}


?>
<!DOCTYPE html>
<html>
<head>
	<title>Query Project</title>
	<meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
<link rel="stylesheet" type="text/css" href="query_interface_style.css">


<!-- Other head content -->

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>

</head>
<body>
    
<div class="col-sm-12">
			<h1 style="text-align: center;">Query Oriented Data Clean</h1>
		</div>
    <div class="b1">
    <button id="toggleButton" class="button1">Click to Display Dataset</button>
    <button id="answerQueryButton" class="button1">Answer Query Question</button>
    </div>
    <br>
    
    <br>
<div class="container">
	<div class="row">
		
    <div class="custom-table">
		<div class="col-sm-12">
        <table class="common-table">
    <thead>
        <tr>
            <th>YEAR</th>
            <th>HOST</th>
            <th>CHAMPION</th>
            <th>RUNNER UP</th>
            <th>THIRD PLACE</th>
            <th>MATCHES PLAYED</th>
            <th>GOALS SCORED</th>
        </tr>
    </thead>
    <?php 
    // Your database connection and data retrieval code here
    echo '<tbody>'; // Start table body
    while($row1 = mysqli_fetch_array($result1)):;?>
        <tr>
            <td<?php echo (empty($row1[0]) ? ' class="empty-cell"' : ''); ?>><?php echo $row1[0];?></td>
            <td<?php echo (empty($row1[1]) ? ' class="empty-cell"' : ''); ?>><?php echo $row1[1];?></td>
            <td<?php echo (empty($row1[2]) ? ' class="empty-cell"' : ''); ?>><?php echo $row1[2];?></td>
            <td<?php echo (empty($row1[3]) ? ' class="empty-cell"' : ''); ?>><?php echo $row1[3];?></td>
            <td<?php echo (empty($row1[4]) ? ' class="empty-cell"' : ''); ?>><?php echo $row1[4];?></td>
            <td<?php echo (empty($row1[5]) ? ' class="empty-cell"' : ''); ?>><?php echo $row1[5];?></td>
            <td<?php echo (empty($row1[6]) ? ' class="empty-cell"' : ''); ?>><?php echo $row1[6];?></td>
        </tr>
    <?php
    endwhile;
    echo '</tbody>'; // End table body
    ?>
</table>

		</div>
	</div>
</div>
</div>
<!-- Modal for displaying the question -->
<div id="questionContainer" style="display: none; text-align: center; margin-top: 20px;">
    <p id="questionText"></p>
    <input type="text" id="userAnswer">
    <button id="submitAnswer">Submit Answer</button>
</div>
<script>
        document.addEventListener("DOMContentLoaded", function () {
            var toggleButton = document.getElementById("toggleButton");
            var errorModal = document.getElementById("errorModal");
            var modifyDataButton = document.getElementById("modifyDataButton");
            var noModifyDataButton = document.getElementById("noModifyDataButton");
            var buttonsContainer = document.getElementById("buttonsContainer");
            var questionContainer = document.getElementById("questionContainer");

            buttonsContainer.style.display = "none"; // Initially hide buttons
            questionContainer.style.display = "none"; // Initially hide question container

            // Function to show the modal
            function showModal() {
                errorModal.style.display = "block";
            }

            // Function to close the modal
            function closeModal() {
                errorModal.style.display = "none";
            }

            // Function to show the buttons
            function showButtons() {
                buttonsContainer.style.display = "block";
            }

            // Function to hide the buttons
            function hideButtons() {
                buttonsContainer.style.display = "none";
            }

            // Function to show the question container
            function showQuestion() {
                questionContainer.style.display = "block";
            }

            // Function to hide the question container
            function hideQuestion() {
                questionContainer.style.display = "none";
            }

            // Check if there's an error message
            var errorMessage = document.querySelector(".error-message");
            if (errorMessage) {
                // Check if the error message contains the specific text
                if (errorMessage.textContent.includes("The \"year\" column can't be created as a primary key")) {
                    showButtons();
                    showQuestion(); // Show the question container
                } else {
                    hideButtons();
                    hideQuestion(); // Hide the question container
                }
            } else {
                hideButtons();
                hideQuestion(); // Hide the question container
            }

            toggleButton.addEventListener("click", function () {
                // Toggle the table's display property
                var table = document.querySelector(".custom-table");
                table.style.display = table.style.display === "none" ? "table" : "none";
            });

            // Event listener for the form submission (assuming you have a form with id "queryForm")
            document.getElementById("queryForm").addEventListener("submit", function (event) {
                var sql_query = document.getElementsByName("sql_query")[0].value;
                if (sql_query.toLowerCase().includes("alter table db_messy add primary key (year)")) {
                    // Create buttons and append them to the error modal
                    var modifyButton = document.createElement("button");
                    modifyButton.id = "modifyDataButton";
                    modifyButton.textContent = "Yes, Modify";

                    var noModifyButton = document.createElement("button");
                    noModifyButton.id = "noModifyDataButton";
                    noModifyButton.textContent = "No, Modify";

                    buttonsContainer.innerHTML = ''; // Clear any existing buttons
                    buttonsContainer.appendChild(modifyButton);
                    buttonsContainer.appendChild(noModifyButton);

                    // Display the modal when the specific query is detected
                    showModal();

                    // Handle the 'Yes, Modify' button click
                    modifyButton.addEventListener("click", function () {
                        console.log("Yes, Modify button clicked");
                        // Redirect to a page where the data modification can be performed
                        window.location.href = "query_question.php"; // Change to the actual modification page
                    });

                    // Handle the 'No, Modify' button click
                    noModifyButton.addEventListener("click", function () {
                        // Close the modal
                        closeModal();
                    });

                    // Show the question container
                    showQuestion();
                    // Hide the buttons container
                    hideButtons();

                    // Prevent the default form submission
                    event.preventDefault();
                }
            });

            // Close the modal when the close button is clicked
            document.getElementById("closeModal").addEventListener("click", closeModal);

            // Event listener for the question submission
            document.getElementById("submitAnswer").addEventListener("click", function () {
                // Get the user's answer
                var userAnswer = document.getElementById("userAnswer").value;

                // Perform any validation on the answer if needed

                // Add the user's answer to the database based on the question
                // Assuming you have a PHP script to handle database updates
                var updateScript = "update_database.php"; // Change to the actual script
                var formData = new FormData();
                formData.append('year_answer', userAnswer);

                fetch(updateScript, {
                    method: 'POST',
                    body: formData
                })
                    .then(response => response.json())
                    .then(data => {
                        // Handle the response, if needed
                        console.log(data);
                    })
                    .catch(error => {
                        console.error('Error:', error);
                    });
            });
        });
    </script>
<!-- This is the code to input user query -->
<form method="post" action="base_interface.php">
    <input type="text" name="sql_query" placeholder="Enter SQL Query">
    <input type="submit" value="Execute Query">
</form>
<div class="container">
<div class="row">
<!-- Display the error message if there is one -->
<?php
if (!empty($error_message)) {
    echo '<p class="error-message">' . $error_message . '</p>';
}
?>
<!-- Buttons container -->
<div id="buttonsContainer">
            <button id="modifyDataButton">Yes, Modify</button>
            <button id="noModifyDataButton">No, Modify </button>
        </div>
<!-- Question container -->
<div id="questionContainer">
    <p id="questionText">Please enter the value for the empty "year" cell:</p>
    <input type="text" id="userAnswer" placeholder="Enter Year">
    <button id="submitAnswer">Submit Answer</button>
    </div>
<!-- Display the query results -->
<?php echo $query_results; ?>
</div>
</div>
<script>
    document.addEventListener("DOMContentLoaded", function () {
        var answerQueryButton = document.getElementById("answerQueryButton");

        answerQueryButton.addEventListener("click", function () {
            // Redirect to your index.html file
            /*  window.location.href = "http://127.0.0.1:5000";   // Change to the actual file path*/
            window.open("http://127.0.0.1:5000", "_blank");

        });
    });
</script>
</body>
</html>