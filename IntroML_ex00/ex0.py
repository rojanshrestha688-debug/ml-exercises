
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Do not alter this path!
IMAGE_PATH: str = "data/Image01.png"


class ImageProcessor:
    def __init__(self, image_path: str, colour_type: str = "BGR"):
        """
        Load and save the provided image, the image colour type and the image directory.
        Use CV2 to load the image.

        Args:
        image_path (str): Path to the input image.
        colour_type (str): Colour type of the image (BGR, RGB, Gray).
        """
        # Extract the parent directory of the image.
        self._image_directory: str = os.path.dirname(image_path)
        if colour_type not in ["BGR", "RGB", "Gray"]:
            raise ValueError("The given colour is not supported!")

        # ToDo: Save the colour type and load the image using CV2.
        self._colour_type: str = colour_type
        self._image: np.ndarray = np.zeros(0)

        self._image = cv2.imread(image_path)

    def get_image_data(self) -> tuple[np.ndarray, str]:
        """
        Return the image data (image and colour scheme).

        Returns:
            tuple(np.ndarray, str): Loaded image and current colour scheme.
        """
        return self._image, self._colour_type

    def show_image(self):
        """
        Show the loaded image using either matplotlib or CV2.
        """

        # ToDo: Show the image depending on the colour type.
        if self._colour_type == "RGB":
            cv2.imshow("Monkey Image", cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR))
        # elif self._colour_type == "Gray":
        #     cv2.imshow("Monkey Image", cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY))
        else:
            cv2.imshow("Monkey Image", self._image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, image_title: str):
        """
        Save the loaded image using either matplotlib or CV2.

        Args:
        image_title (str): Title of the image with the corresponding extension.
        """

        # Combine the image parent directory and the given title to create the path for the new image.
        total_image_path: str = os.path.join(self._image_directory, image_title)

        # ToDo: Save the image.
        cv2.imwrite(total_image_path, self._image)


    def convert_colour(self):
        """
        Convert a colour image from BGR to RGB or vice versa.
        Do not use functions from external libraries.
        Solve this task by using indexing.
        """
        if self._colour_type not in ["RGB", "BGR"]:
            raise ValueError("The function only works for colour images!")

        # ToDo: Perform the colour conversion.
        self._image = self._image[:,:,::-1]

        # ToDo: Update the colour type.
        self._colour_type = self._colour_type[::-1]

    def clip_image(self, clip_min: int, clip_max: int):
            """
            Clip all colour values in the image to a given min and max value.
            Do not use functions from external libraries.
            Solve this task by using indexing.

            Args:
            clip_min (int): Minimum image colour intensity.
            clip_max (int): Maximum image colour intensity.
            """
            # ToDo: Clip the image values to the given values.
            self._image[self._image < clip_min] = clip_min
            self._image[self._image > clip_max] = clip_max

    def convert_to_grayscale(self, method: str = "`lightness`"):
        """
        Convert a colour image to a grayscale image.
        Write the different options from scratch.

        Args:
        method (str): Method for the colour conversion, either lightness, average or luminosity.
        """
        if method not in ["lightness", "average", "luminosity"]:
            raise ValueError("The given method is not supported!")
        if self._colour_type not in ["BGR", "RGB"]:
            raise ValueError("The function only works for colour images!")

        if self._colour_type == "BGR":
            b = self._image[:, :, 0]
            g = self._image[:, :, 1]
            r = self._image[:, :, 2]
        else:  # if its not in BGR then its in RGB 
            r = self._image[:, :, 0]
            g = self._image[:, :, 1]
            b = self._image[:, :, 2]

        if method == "lightness":
            max_val = np.maximum(np.maximum(r, g), b)
            min_val = np.minimum(np.minimum(r, g), b)
            gray = (max_val + min_val) / 2

        elif method == "average":
            gray = (r + g + b) / 3

        elif method == "luminosity":
            gray = 0.21 * r + 0.72 * g + 0.07 * b

        self._image = gray.astype(np.uint8)

        # ToDo: Update colour type
        self._colour_type = "Gray"



    def rotate_image(self, degrees: int = 0):
        """
        Rotate an image by a given angle (k * 90) clockwise.
        Do not use functions from external libraries apart from numpy.transpose.

        Args:
        degrees (int): Rotation angle.
        """
        if degrees % 90 != 0:
            raise ValueError("The provided rotation angle must be a multiple of 90!")

        # ToDo: Rotate the image depending on the given rotation value.
        rotation_value = (degrees // 90) % 4
        if rotation_value == 0:
            return
        elif rotation_value == 1:
            self._image = np.transpose(self._image, axes=(1, 0,2))[:,::-1,:]
        elif rotation_value == 2:
            self._image = self._image[::-1,::-1,:]
        elif rotation_value == 3:
            self._image = np.transpose(self._image, axes=(1, 0,2))[::-1,:,:]

    def flip_image(self, flip_value: int):
        """
        Flip an image either horizontally (0), vertically (1) or both ways (2).
        Do not use functions from external libraries.

        Args:
        flip_value (int): Value to determine how the image should be flipped.
        """
        if flip_value not in [0, 1, 2]:
            raise ValueError("The provided flip value must be either 0, 1 or 2!")

        # ToDo: Flip the image using indexing.
        if flip_value == 0:
            self._image = self._image[:,::-1,:]
        elif flip_value == 1:
            self._image = self._image[::-1,:,:]
        else:
            self._image = self._image[::-1,::-1,:]

    def crop_center(self, new_height: int, new_width: int):
        """
        Crop the image to a given size around the center.
        Do not use functions from external libraries.

        Args:
        new_height (int): Height of the cropped image.
        new_width (int): Width of the cropped image.
        """
        # ToDo: Check that the given parameters are valid!
        height, width = self._image.shape[:2]
        if new_height <= 0 or new_width <= 0:
            raise ValueError("value should be positive and greater then 0")

        if height < new_height or width < new_width:
            raise ValueError("The provided are not valid")

        # ToDo: Crop the image around the center.
        left = (width - new_width) // 2
        right = (width + new_width) // 2
        top = (height - new_height) // 2
        bottom = (height + new_height) // 2

        self._image = self._image[top:bottom,left:right]

    def resize_image(self, new_height: int, new_width: int):
        """
        Resize an image to an arbitrary size using CV2.

        Args:
        new_height (int): Height of the resized image.
        new_width (int): Width of the resized image.
        """
        # ToDo: Resize the image. Research the available options in CV2.
        self._image =cv2.resize(self._image,(new_height, new_width))


if __name__ == '__main__':
    processor = ImageProcessor(image_path=IMAGE_PATH, colour_type="BGR")
    # processor.convert_colour("")
    # processor.clip_image(5,200)
    # processor.convert_to_grayscale("lightness")
    # processor.show_image()
    # processor.convert_to_grayscale("average")
    # processor.show_image()
    # processor.convert_to_grayscale("luminosity")
    # processor.rotate_image(90)
    # processor.flip_image(1)
    # processor.crop_center(100,100)
    # processor.resize_image(100,100)
    processor.show_image()






